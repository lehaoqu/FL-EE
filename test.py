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

sum = 0.0
for i in range(10):
    sum += i**2
print(sum/10)