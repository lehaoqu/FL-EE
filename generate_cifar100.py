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
from dataset.cifar100_dataset import CIFARClassificationDataset
from sklearn.model_selection import train_test_split

random.seed(1)
np.random.seed(1)
num_clients = 1200
dir_path = "dataset/cifar100-224-d03-1200/"
train_ratio = 0.8

# Allocate data to users
def generate_cifar100(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/valid data for clients
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    valid_path = dir_path + "valid/"


    if check(config_path, train_path, valid_path, num_clients, niid, balance, partition):
        return
       
    # == save train valid for clients  ==
    # Get Cifar100 data
    transform = transforms.Compose([transforms.Resize(224),
    # transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                                        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                                    ),
                                    ])

    # trainset = torchvision.datasets.CIFAR100(
    #     root=dir_path+"rawdata", train=True, download=True)
    
    global_train_path = dir_path + "rawdata/cifar-100-python/train"
    global_trainset = load_pkl(global_train_path)
    
    origin_train_len = len(global_trainset)
    train_len = int(origin_train_len * train_ratio)
    valid_len = origin_train_len - train_len
    
    train_dataset = CIFARClassificationDataset(path=global_train_path)

    
    
    
    # validset = torch.utils.data.Subset(trainset, list(range(train_len, origin_train_len)))
    # trainset = torch.utils.data.Subset(trainset, list(range(train_len)))
    
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=train_len, shuffle=False)
    # validloader = torch.utils.data.DataLoader(
    #     validset, batch_size=valid_len, shuffle=False)
    

    # for idx, train_data in enumerate(trainloader, 0):
    #     trainset.data, trainset.targets = train_data
    # for idx, valid_data in enumerate(validloader, 0):
    #     validset.data, validset.targets = valid_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(train_dataset.pixel_values)
    dataset_label.extend(train_dataset.labels)

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
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar100(dir_path, num_clients, niid, balance, partition)