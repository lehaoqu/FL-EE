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
import torch

def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        valid_data_dir = os.path.join('dataset', dataset, 'valid/')

        valid_file = valid_data_dir + str(idx) + '.npz'
        with open(valid_file, 'rb') as f:
            valid_data = np.load(f, allow_pickle=True)['data'].tolist()

        return valid_data


def read_client_data(dataset, idx, is_train=True):
    if "News" in dataset:
        return read_client_data_text(dataset, idx, is_train)
    elif "Shakespeare" in dataset:
        return read_client_data_Shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        valid_data = read_data(dataset, idx, is_train)
        X_valid = torch.Tensor(valid_data['x']).type(torch.float32)
        y_valid = torch.Tensor(valid_data['y']).type(torch.int64)
        valid_data = [(x, y) for x, y in zip(X_valid, y_valid)]
        return valid_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        valid_data = read_data(dataset, idx, is_train)
        X_valid, X_valid_lens = list(zip(*valid_data['x']))
        y_valid = valid_data['y']

        X_valid = torch.Tensor(X_valid).type(torch.int64)
        X_valid_lens = torch.Tensor(X_valid_lens).type(torch.int64)
        y_valid = torch.Tensor(valid_data['y']).type(torch.int64)

        valid_data = [((x, lens), y) for x, lens, y in zip(X_valid, X_valid_lens, y_valid)]
        return valid_data


def read_client_data_Shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        valid_data = read_data(dataset, idx, is_train)
        X_valid = torch.Tensor(valid_data['x']).type(torch.int64)
        y_valid = torch.Tensor(valid_data['y']).type(torch.int64)
        valid_data = [(x, y) for x, y in zip(X_valid, y_valid)]
        return valid_data

