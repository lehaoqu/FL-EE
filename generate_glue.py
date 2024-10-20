import numpy as np
import os
import sys
import random
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import pickle

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from dataset.utils.dataset_utils import load_tsv, separate_data, save_file, check
from dataset.glue_dataset import GLUEClassificationDataset

random.seed(1)
np.random.seed(1)
num_clients = 120
dir_path = "dataset/glue/"
train_ratio = 0.8


def generate_glue(dir_path, task, num_clients, niid, balance, partition, tokenizer):
    train_dir = dir_path + task + "/train/"
    valid_dir = dir_path + task + "/valid/"
    config_path = dir_path + task + "/config.json"
    
    if check(config_path, train_dir, valid_dir, num_clients, niid, balance, partition):
        return
    
    train_dir = os.path.join(train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    valid_dir = os.path.join(valid_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
        
    global_train_path = dir_path + task + '/train.tsv'
    
    global_trainset = load_tsv(global_train_path)
    train_len = int(len(global_trainset) * train_ratio)
    valid_len = len(global_trainset) - train_len
    # train_set = global_trainset[:train_len]
    # valid_set = global_trainset[:train_len]
    
    train_dataset = GLUEClassificationDataset(path=global_train_path, tokenizer=tokenizer, need_process=True)

    
    num_classes = 2
    print(f'Number of classes: {num_classes}')
    
    dataset_input_id = []
    dataset_label = []
    dataset_mask = []
    
    dataset_input_id.extend(np.vstack([t.cpu().numpy() for t in train_dataset.input_ids]))
    # dataset_input_id.extend(np.vstack([t.cpu().numpy() for t in valid_dataset.input_ids]))
    dataset_label.extend(np.vstack([t.cpu().numpy() for t in train_dataset.labels]))
    # dataset_label.extend(np.vstack([t.cpu().numpy() for t in valid_dataset.labels]))
    dataset_mask.extend(np.vstack([t.cpu().numpy() for t in train_dataset.attention_mask]))
    # dataset_mask.extend(np.vstack([t.cpu().numpy() for t in valid_dataset.attention_mask]))
    dataset_input_id = np.array(dataset_input_id)
    dataset_label = np.array(dataset_label)
    dataset_mask = np.array(dataset_mask)
    
    input_ids, labels, attention_mask, statistic = separate_data((dataset_input_id, dataset_label), num_clients, num_classes,
                            niid, balance, partition, class_per_client=2, attention_mask=dataset_mask)
    
    # == split train and valid ==
    train_data, valid_data = [], []
    for i in range(len(labels)):
        
        
        train_input_ids, valid_input_ids, train_attention_mask, valid_attention_mask, train_labels, valid_labels = train_test_split(
            input_ids[i], attention_mask[i], labels[i], train_size=train_ratio, shuffle=True
        )
        
        # train_attention_mask, valid_attention_mask, _, _ = train_test_split(
        #     attention_mask[i], labels[i], train_size=train_ratio, shuffle=True
        # )
        
        train_data.append({'input_ids': train_input_ids, 'attention_mask': train_attention_mask, 'labels': train_labels})
        valid_data.append({'input_ids': valid_input_ids, 'attention_mask': valid_attention_mask, 'labels': valid_labels})
                
    
    save_file(config_path, train_dir, valid_dir, train_data, valid_data, num_clients, num_classes, 
    statistic, niid, balance, partition)
    
    # == test ==
    test_path = dir_path + task + '/test.tsv'
    test_dataset = GLUEClassificationDataset(path=test_path, tokenizer=tokenizer, need_process=True)
    dataset_input_id = []
    dataset_label = []
    dataset_mask = []
    
    dataset_input_id.extend(np.vstack([t.cpu().numpy() for t in test_dataset.input_ids]))
    dataset_label.extend(np.vstack([t.cpu().numpy() for t in test_dataset.labels]))
    dataset_mask.extend(np.vstack([t.cpu().numpy() for t in test_dataset.attention_mask]))
    dataset_input_id = np.array(dataset_input_id)
    dataset_label = np.array(dataset_label)
    dataset_mask = np.array(dataset_mask)
    test_data = {'input_ids': dataset_input_id, 'attention_mask': dataset_mask, 'labels': dataset_label}
    test_pkl_path = dir_path + task + '/test.pkl'
    with open(test_pkl_path, 'wb') as f:
            pickle.dump(test_data, f)
        
    print("Total number of samples:", train_len + valid_len)
    print("The number of train samples:", train_len)
    print("The number of valid samples:", valid_len)
    print()
        
        
if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    tokenizer = AutoTokenizer.from_pretrained(
            'models/google-bert/bert-12-uncased',
            padding_side="right",
            model_max_length=128,
            use_fast=False,
        )
    # for task in ['sst2']:
    # for task in ['sst2', 'mrpc', 'qqp', 'qnli']:
    for task in ['qqp']:
        generate_glue(dir_path, task, num_clients, niid, balance, partition, tokenizer)