import torch

import torch.nn as nn
from typing import *


class Policy():
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.exits_num = self.args.exits_num
        self.loss_func = nn.CrossEntropyLoss()
    
    def __call__(self, exits_logits, label, ws=None) -> torch.tensor:
        assert self.exits_num == len(exits_logits), f'expected {self.exits_num}, but {len(exits_logits)}'
        
        ws = [1 for _ in range(self.exits_num)] if ws is None else ws
                    
        loss = torch.zeros(1).to(self.device)
        for i, exit_logits in enumerate(exits_logits):
            loss += self.loss_func(exit_logits, label) * ws[i]
        return loss

    def eval(self, exits_logits:List[torch.tensor]):
        return exits_logits