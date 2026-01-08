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
import argparse
from datasets import load_dataset
import csv

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from dataset.utils.dataset_utils import load_tsv, separate_data, save_file, check
from dataset.glue_dataset import GLUEClassificationDataset

random.seed(1)
np.random.seed(1)
num_clients = 100
train_ratio = 0.8

def download_glue_task(task, dir_path):
    """Download GLUE task dataset from HuggingFace and save as TSV files"""
    task_path = os.path.join(dir_path, task)
    train_file = os.path.join(task_path, 'train.tsv')
    test_file = os.path.join(task_path, 'test.tsv')
    
    # Check if files already exist
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"GLUE {task} dataset already exists at {task_path}")
        return
    
    print(f"Downloading GLUE {task} dataset...")
    os.makedirs(task_path, exist_ok=True)
    
    # Map task names to GLUE dataset names
    task_map = {
        'sst2': ('glue', 'sst2'),
        'mrpc': ('glue', 'mrpc'),
        'qqp': ('glue', 'qqp'),
        'qnli': ('glue', 'qnli'),
        'rte': ('glue', 'rte'),
        'wnli': ('glue', 'wnli')
    }
    
    if task not in task_map:
        raise ValueError(f"Unsupported task: {task}")
    
    dataset_name, config_name = task_map[task]
    
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name, config_name)
    
    # Save train split
    train_data = dataset['train']
    with open(train_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write header
        writer.writerow(train_data.column_names)
        # Write data
        for example in train_data:
            writer.writerow([example[col] for col in train_data.column_names])
    
    # Save test/validation split (use validation as test for GLUE)
    test_split = 'validation' if 'validation' in dataset else 'test'
    test_data = dataset[test_split]
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        # Write header
        writer.writerow(test_data.column_names)
        # Write data
        for example in test_data:
            writer.writerow([example[col] for col in test_data.column_names])
    
    print(f"GLUE {task} dataset downloaded to {task_path}")
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Test samples: {len(test_data)}")



def generate_glue(dir_path, task, num_clients, niid, balance, partition, tokenizer, alpha=1000):
    # Set alpha in dataset_utils
    import dataset.utils.dataset_utils as dataset_utils
    dataset_utils.alpha = alpha
    print(f"Using alpha: {alpha}")
    
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
        
    # Download GLUE task dataset if not exists
    download_glue_task(task, dir_path)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('niid_type', choices=['iid', 'noniid'])
    parser.add_argument('balance_type', choices=['balance', 'unbalanced'])
    parser.add_argument('partition')
    parser.add_argument('--alpha', type=float, default=1000, help='Alpha parameter for Dirichlet distribution')
    parser.add_argument('--task', type=str, default='all', 
                       choices=['all', 'sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli'],
                       help='GLUE task to generate (default: all)')
    args = parser.parse_args()
    
    niid = args.niid_type == "noniid"
    balance = args.balance_type == "balance"
    partition = args.partition if args.partition != "-" else None
    
    # Generate dir_path based on alpha
    if niid:
        alpha_str = str(args.alpha).rstrip('0').rstrip('.') if '.' in str(args.alpha) else str(int(args.alpha))
        dir_path = f"dataset/glue_noniid{alpha_str}/"
    else:
        dir_path = "dataset/glue/"

    tokenizer = AutoTokenizer.from_pretrained(
            'models/google/bert_uncased_L-12_H-128_A-2',
            padding_side="right",
            model_max_length=128,
            use_fast=False,
        )
    
    # Determine which tasks to generate
    if args.task == 'all':
        tasks = ['sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli']
    else:
        tasks = [args.task]
    
    for task in tasks:
        generate_glue(dir_path, task, num_clients, niid, balance, partition, tokenizer, args.alpha)