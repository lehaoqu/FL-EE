import torch

from dataset import (
    get_cifar_dataset,
    get_glue_dataset
)

GLUE = ['douban', 'cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']


CIFAR = ['cifar100-224-d03']


def load_dataset_loader(args, file_name=None, id=None, eval_valids=False):
    if args.dataset in CIFAR:
        if eval_valids:
            dataset = get_cifar_dataset(args=args, path=f'dataset/{args.dataset}/valid/', eval_valids=eval_valids)
        else:
            if file_name == 'test':
                dataset = get_cifar_dataset(args=args, path=f'dataset/{args.dataset}/rawdata/cifar-100-python/{file_name}')
            else:
                dataset = get_cifar_dataset(args=args, path=f'dataset/{args.dataset}/{file_name}/{id}.pkl')
            
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=None)
        return dataset, dataset_loader
    
    if args.dataset in GLUE:
        if eval_valids:
            dataset = get_glue_dataset(args=args, path=f'dataset/glue/{args.dataset}/valid/', eval_valids=eval_valids)
        else:
            if file_name == 'test':
                dataset = get_glue_dataset(args=args, path=f'dataset/glue/{args.dataset}/{file_name}.tsv')
            else:
                dataset = get_glue_dataset(args=args, path=f'dataset/glue/{args.dataset}/{file_name}/{id}.pkl')        
        
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True, collate_fn=None)
        return dataset, dataset_loader
        