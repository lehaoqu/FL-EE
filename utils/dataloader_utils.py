import torch

from dataset import (
    get_cifar_dataset,
    get_glue_dataset,
    get_svhn_dataset,
    get_imagenet_dataset
)

GLUE = ['douban', 'cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']


CIFAR = 'cifar100'
IMAGENET = 'imagenet'

SVHN = 'svhn'


def load_dataset_loader(args, file_name=None, id=None, eval_valids=False, shuffle=True):
    if SVHN in args.dataset:
        if eval_valids:
            dataset = get_svhn_dataset(args=args, path=f'dataset/{args.dataset}/valid/', eval_valids=eval_valids)
        else:
            if file_name == 'test':
                dataset = get_svhn_dataset(args=args, path=f'dataset/{args.dataset}/{file_name}_32x32.mat')
            else:
                dataset = get_svhn_dataset(args=args, path=f'dataset/{args.dataset}/{file_name}/{id}.pkl')
            
    elif CIFAR in args.dataset:
        if eval_valids:
            dataset = get_cifar_dataset(args=args, path=f'dataset/{args.dataset}/valid/', eval_valids=eval_valids)
        else:
            if file_name == 'test':
                dataset = get_cifar_dataset(args=args, path=f'dataset/{args.dataset}/rawdata/cifar-100-python/{file_name}')
            else:
                dataset = get_cifar_dataset(args=args, path=f'dataset/{args.dataset}/{file_name}/{id}.pkl')
    elif IMAGENET in args.dataset:
        if eval_valids:
            dataset = get_imagenet_dataset(args=args, path=f'dataset/{args.dataset}/valid/', eval_valids=eval_valids)
        else:
            if file_name == 'test':
                dataset = get_imagenet_dataset(args=args, path=f'dataset/{args.dataset}/{file_name}')
            else:
                dataset = get_imagenet_dataset(args=args, path=f'dataset/{args.dataset}/{file_name}/{id}.pkl')
    
         
    elif args.dataset in GLUE:
        if eval_valids:
            dataset = get_glue_dataset(args=args, path=f'dataset/glue/{args.dataset}/valid/', eval_valids=eval_valids)
        else:
            if file_name == 'test':
                dataset = get_glue_dataset(args=args, path=f'dataset/glue/{args.dataset}/test.pkl')
            else:
                dataset = get_glue_dataset(args=args, path=f'dataset/glue/{args.dataset}/{file_name}/{id}.pkl')        
       
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=shuffle, collate_fn=None)
    return dataset, dataset_loader
        