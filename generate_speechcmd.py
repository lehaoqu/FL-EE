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
import pickle
import torchvision.transforms as transforms
from dataset.utils.dataset_utils import check, separate_data, split_data, save_file, save_origin_file, load_pkl
from dataset.speechcmd_dataset import SPEECHCOMMANDSDATASET
from sklearn.model_selection import train_test_split
import argparse

random.seed(1)
np.random.seed(1)
num_clients = 100
train_ratio = 0.8

# Allocate data to users
def generate_speechcommands(dir_path, num_clients, niid, balance, partition, test, alpha=1000):
    # Set alpha in dataset_utils
    import dataset.utils.dataset_utils as dataset_utils
    dataset_utils.alpha = alpha
    print(f"Using alpha: {alpha}")
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    
    if test:
        test_set = SPEECHCOMMANDSDATASET(root='./dataset/speechcmds', type='testing')
        dataset_image = test_set.pixel_values
        dataset_label = test_set.labels
        dataset_image = np.array([t.numpy() for t in dataset_image])
        dataset_label = np.array(dataset_label)
        dct = {b'data': dataset_image, b'fine_labels': dataset_label}
        with open(dir_path + 'test', 'wb') as f:
            pickle.dump(dct, f)
        print('test pkl save to disk')
        return
      
    # Setup directory for train/valid data for clients
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    valid_path = dir_path + "valid/"


    if check(config_path, train_path, valid_path, num_clients, niid, balance, partition):
        return
       
    # == save train valid for clients  ==
    
    train_set = SPEECHCOMMANDSDATASET(root='./dataset/speechcmds', type='training')
    valid_set = SPEECHCOMMANDSDATASET(root='./dataset/speechcmds', type='validation')
    
    origin_train_len = len(train_set)
    train_len = int(origin_train_len * train_ratio)

    dataset_image = []
    dataset_label = []

    dataset_image.extend(train_set.pixel_values)
    dataset_label.extend(train_set.labels)
    dataset_image.extend(valid_set.pixel_values)
    dataset_label.extend(valid_set.labels)

    first_shape = dataset_image[0].shape
    all_same_shape = all(t.shape == first_shape for t in dataset_image[1:])
    print("All tensors have the same shape:", all_same_shape)

    dataset_image = np.array([t.numpy() for t in dataset_image])
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

        train_data.append({b'data': X_train, b'fine_labels': y_train})
        valid_data.append({b'data': X_valid, b'fine_labels': y_valid})

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
        dir_path = f"dataset/speechcmds_noniid{alpha_str}/"
    else:
        dir_path = "dataset/speechcmds/"
    
    print("Generating training data...")
    generate_speechcommands(dir_path, num_clients, niid, balance, partition, False, args.alpha)
    
    print("Generating test data...")
    generate_speechcommands(dir_path, num_clients, niid, balance, partition, True, args.alpha)