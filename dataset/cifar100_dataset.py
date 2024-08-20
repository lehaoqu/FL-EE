import torch
import numpy as np
from torchvision import transforms

from torch.utils.data import Dataset
from dataset.utils.dataset_utils import load_pickle


import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class CIFARClassificationDataset(Dataset):
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            ),
        ])
    

    def transform_for_vit(self):
        images = self.ann[b'data']
        images_reshaped = np.reshape(images, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)/255
        images_transformed = [self.transform(image_reshape) for image_reshape in images_reshaped]
        images_transformed = np.stack(images_transformed, axis=0, dtype=np.float32)
        self.ann[b'data'] = images_transformed
    
    
    def __init__(self, args=None, path=None, ann=None, is_valid=None):
        self.path = path
        self.ann:dict = load_pickle(self.path)
        if is_valid is not None:
            if is_valid is True:
                self.ann:dict = {key: value[int((1-args.valid_ratio)*len(value)):] if isinstance(value, list) or isinstance(value, np.ndarray) else value for key, value in self.ann.items()}
        # self.transform_for_vit()
        
    def __len__(self):
        return len(self.ann[b'fine_labels'])
    
    def __getitem__(self, index):
        image = self.ann[b'data'][index]
        
        image_reshape = np.reshape(image, (3, 32, 32)).transpose(1, 2, 0)/255
        image_transformed = self.transform(image_reshape)
        return dict(
            pixel_values=image_transformed,
            labels=self.ann[b'fine_labels'][index],
        )
        return dict(
            pixel_values=image,
            labels=self.ann[b'fine_labels'][index],
        )
        

# # # 数据预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224, 224)),
#     transforms.Normalize(
#         (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
#         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
#     ),
# ])



# # 加载 CIFAR-100 数据集
# trainset = torchvision.datasets.CIFAR100(root='./res/datasets/cifar100', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
# print(trainloader)

# trainset = torchvision.datasets.CIFAR100(root='./res/datasets/cifar100', train=True, download=False)
# x = trainset.data
# print(x[-1].shape)
# print(np.mean(x[-1]/255, axis=(0, 1)))
# y = transform(x[-1])
# print(y.shape)
# print(torch.mean(y, axis=(1, 2)))
# trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
# print(trainloader)


# trainset = CIFARClassificationDataset(path='res/datasets/cifar100/cifar-100-python/train_non-iid_0')