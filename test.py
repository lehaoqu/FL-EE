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

dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank()

for i in range(100):
    t = torch.rand((32,197))
    s = torch.rand((32,197))
    print(angle_criterion(s, t))