import torch
import torch.nn as nn
import random

from typing import *
from trainer.baseHFL import BaseServer, BaseClient, GLUE
from trainer.generator.generator import Generator_LATENT, Generator_CIFAR
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, AdamW

import torch
import torch.nn.functional as F

t = torch.zeros(4)

print(t)