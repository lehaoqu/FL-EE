import torch
import torch.nn as nn
import random

from typing import *
from trainer.baseHFL import BaseServer, BaseClient, GLUE
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, AdamW

import torch
import torch.nn.functional as F

import torch

# 假设你有一个张量
tensor = torch.tensor([0.1, 0.3, 0.6, 0.8]).to(0)

# 找到最大值
max_value = tensor.max()

# 将最大值设为1，其他设为0
result = torch.where(tensor == max_value, torch.tensor(1), torch.tensor(0))

print(result)