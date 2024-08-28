import torch

import torch.nn as nn
from typing import *

def add_args(parser):
    return parser

class Policy():
    def __init__(self, args):
        self.name = 'base'
        self.args = args
        self.device = self.args.device
        self.exits_num = self.args.exits_num
        self.loss_func = nn.CrossEntropyLoss()
    
    def train(self, model, batch, label, ws=None) -> torch.tensor:
        exits_logits = model(**batch)
        
        assert self.exits_num == len(exits_logits), f'expected {self.exits_num}, but {len(exits_logits)}'
        
        ws = [1 for _ in range(self.exits_num)] if ws is None else ws
                    
        exits_loss = ()
        for i, exit_logits in enumerate(exits_logits):
            exits_loss += (self.loss_func(exit_logits, label) * ws[i],)
        return exits_loss, exits_logits

    def train_all_logits(self, exits_logits):
        return exits_logits

    def __call__(self, exits_logits):
        return exits_logits