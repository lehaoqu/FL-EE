import torch
import numpy as np
from torchvision import transforms

from torch.utils.data import Dataset
from dataset.utils.dataset_utils import load_np, load_pkl


import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class CIFARClassificationDataset(Dataset):
        
    
    def transform_for_vit(images: torch.tensor):

        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ])
        
        images = images.float()
        images_reshaped = images.view(-1, 3, 32, 32) / 255.0
        return torch.stack([transform(image) for image in images_reshaped], dim=0)
        # return ((imaged_transformed-mean.view(1,3,1,1))/std.view(1,3,1,1))
    
    
    def generator_transform_tensor(images: torch.tensor):
        
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ])
        images_reshaped = images.view(-1, 3, 32, 32)
        return torch.stack([transform(image) for image in images_reshaped], dim=0)

    
    def __init__(self, args=None, path=None, eval_valids=False):
        self.path = path
        if eval_valids:
            dict_all = [load_pkl(f'{path}{i}.pkl') for i in range(args.total_num)]
            total_data = {}
            for key in dict_all[0].keys():
                for dic in dict_all:
                    total_data.setdefault(key, []).extend(dic[key])
            self.ann = total_data
            
        else:
            self.ann = load_pkl(path)
            
            self.ann[b'data'] = [torch.tensor(row, dtype=torch.float32) for row in self.ann[b'data']]
            self.ann[b'fine_labels'] = [torch.tensor(row, dtype=torch.long) for row in self.ann[b'fine_labels']]
            
        self.pixel_values = self.ann[b'data']
        self.labels = self.ann[b'fine_labels']
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return dict(
            pixel_values=self.pixel_values[index],
            labels=self.labels[index],
        )
        
