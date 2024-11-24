
from dataset.utils.dataset_utils import load_tsv, load_np, load_pkl
from dataset import get_imagenet_dataset
import torch
from dataset.svhn_dataset import SVHNClassificationDataset

args = {'total_num': 100}
dataset = get_imagenet_dataset(args=args, path=f'dataset/cifar100_noniid1000/train/0.pkl')
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=None)
for idx, data in enumerate(dataset_loader):
    for key in data.keys():
        print(data[key])

        
