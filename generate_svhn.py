# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file, save_origin_file, load_pkl
from dataset.svhn_dataset import SVHNClassificationDataset
from sklearn.model_selection import train_test_split
import scipy.io as sio
import argparse
import urllib.request
from tqdm import tqdm

random.seed(1)
np.random.seed(1)
num_clients = 100
train_ratio = 0.8

def download_file(url, dest_path):
    """Download file from URL with progress bar"""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)
    
    print(f"Download completed: {dest_path}")

# Allocate data to users
def generate_svhn(dir_path, num_clients, niid, balance, partition, alpha=1000):
    # Set alpha in dataset_utils
    import dataset.utils.dataset_utils as dataset_utils
    dataset_utils.alpha = alpha
    print(f"Using alpha: {alpha}")
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/valid data for clients
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    valid_path = dir_path + "valid/"


    if check(config_path, train_path, valid_path, num_clients, niid, balance, partition):
        return
       
    # == save train valid for clients  ==
    # Download SVHN data if not exists
    train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    
    global_train_path = dir_path + "train_32x32.mat"
    global_test_path = dir_path + "test_32x32.mat"
    
    download_file(train_url, global_train_path)
    download_file(test_url, global_test_path)
    
    global_trainset = sio.loadmat(global_train_path)
    
    origin_train_len = global_trainset['X'].shape[-1]
    train_len = int(origin_train_len * train_ratio)
    valid_len = origin_train_len - train_len
    
    train_dataset = SVHNClassificationDataset(path=global_train_path)

    dataset_image = []
    dataset_label = []

    dataset_image.extend(train_dataset.pixel_values)
    dataset_label.extend(train_dataset.labels)

    first_shape = dataset_image[0].shape
    all_same_shape = all(t.shape == first_shape for t in dataset_image[1:])
    print("All tensors have the same shape:", all_same_shape)
    

    dataset_image = np.array([t.numpy() for t in dataset_image])
    # print(dataset_label)
    dataset_label = np.array(dataset_label)
    

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, class_per_client=2)
    
    train_data, valid_data = [], []
    num_samples = {'train':[], 'valid':[]}

    for i in range(len(y)):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'X': X_train, 'y': y_train})
        valid_data.append({'X': X_valid, 'y': y_valid})

    # print("Total number of samples:", sum(num_samples['train'] + num_samples['valid']))
    # print("The number of train samples:", num_samples['train'])
    # print("The number of valid samples:", num_samples['valid'])
    # print()
    del X, y
    
    print("save train valid sets for clients")
    save_file(config_path, train_path, valid_path, train_data, valid_data, num_clients, num_classes, 
        statistic, niid, balance, partition)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('niid_type', choices=['iid', 'noniid'])
    parser.add_argument('balance_type', choices=['balance', 'unbalanced'])
    parser.add_argument('partition')
    parser.add_argument('--alpha', type=float, default=1000, help='Alpha parameter for Dirichlet distribution')
    args = parser.parse_args()
    
    niid = args.niid_type == "noniid"
    balance = args.balance_type == "balance"
    partition = args.partition if args.partition != "-" else None
    
    # Generate dir_path based on alpha
    if niid:
        alpha_str = str(args.alpha).rstrip('0').rstrip('.') if '.' in str(args.alpha) else str(int(args.alpha))
        dir_path = f"dataset/svhn_noniid{alpha_str}/"
    else:
        dir_path = "dataset/svhn/"

    generate_svhn(dir_path, num_clients, niid, balance, partition, args.alpha)