import torch

import torch.nn as nn
from typing import *


class Policy():
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.exits_num = self.args.exits_num
        self.loss_func = nn.CrossEntropyLoss()
    
    def __call__(self, args) -> torch.tensor:
        exits_logits = args.exits_logits
        assert self.exits_num == len(exits_logits), f'expected {self.exits_num}, but {len(exits_logits)}'
        
        label = args.label
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i, logits in enumerate(exits_logits):
            tmp = logits + pred_ensembels[-1]
            pred_ensembels.append(tmp)
            
        loss = torch.zeros(1).to(self.device)
        for i, logits in enumerate(exits_logits):
            with torch.no_grad:
                pred_ensembel = pred_ensembels[i]
            pred_final = pred_ensembel + logits
            
            loss += self.loss_func(pred_final, label)
        return loss

    def eval(self, former_exits_logits:List[torch.tensor]):
        return torch.sum(former_exits_logits)