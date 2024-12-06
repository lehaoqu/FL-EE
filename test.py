import os
import random
from typing import Tuple
from pathlib import Path

import torch
import numpy as np
from torchvision.transforms import Compose
from torchvision import transforms as tv_transforms
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS, HASH_DIVIDER, EXCEPT_FOLDER, _load_list
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchaudio import transforms, load
from torchaudio.compliance.kaldi import fbank

from collections import defaultdict
import flwr #! DON'T REMOVE -- bad things happen
import pdb

a = torch.tensor([[2.,3.],[4.,5.]])
b = torch.tensor([[0.,1.],[2.,3.]])
print(torch.mean(a-b))