import torch
import torch.nn as nn
import random

from typing import *
from trainer.baseHFL import BaseServer, BaseClient, GLUE
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, AdamW

import torch
import torch.nn.functional as F

# 创建两个张量
tensor1 = torch.tensor([1])  # 假设有5个样本，每个样本3个特征
tensor2 = torch.tensor([1,2,3])

# 计算欧氏距离
euclidean_distance = F.pairwise_distance(tensor1, tensor2)
print("Euclidean Distance:", euclidean_distance)