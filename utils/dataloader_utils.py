import torch

from dataset import (
    get_cifar_dataset,
    get_glue_dataset
)

GLUE_TASKS = ['douban', 'cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']
GLUE_DATASETS = [f"{task}_dataset" for task in GLUE_TASKS]

CIFAR_TASKS = ['cifar100-224-d03']
CIFAR_DATASETS = [f"{task}_dataset" for task in CIFAR_TASKS]

def load_dataset_loader(args, is_valid=True):
    if args.dataset in CIFAR_TASKS:
        file_name = 'train' if is_valid else 'test'
        
        dataset = get_cifar_dataset(args=args, path=f'dataset/{args.dataset}/rawdata/cifar-100-python/{file_name}', is_valid=is_valid)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=None)
        return dataset_loader