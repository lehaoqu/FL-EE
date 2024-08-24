import torch

import torch.nn as nn
from typing import *


class Policy():
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.exits_num = self.args.exits_num
        self.loss_func = nn.CrossEntropyLoss()
    
    def __call__(self, model, batch, label, ws=None) -> torch.tensor:
        batch['policy'] = 'boosted'
        exits_logits = model(**batch)
        
        assert self.exits_num == len(exits_logits), f'expected {self.exits_num}, but {len(exits_logits)}'
        
        ws = [i+1 for i in range(self.exits_num)] if ws is None else ws
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i, logits in enumerate(exits_logits):
            tmp = logits + pred_ensembels[-1]
            pred_ensembels.append(tmp)
            
        loss = torch.zeros(1).to(self.device)
        for i, logits in enumerate(exits_logits):
            with torch.no_grad():
                pred_ensembel = pred_ensembels[i]
            pred_final = pred_ensembel + logits
            
            loss += self.loss_func(pred_final, label) * ws[i]
        return loss

    def eval(self, exits_logits:List[torch.tensor]):
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i in range(self.exits_num):
            tmp = pred_ensembels[-1] + exits_logits[i]
            pred_ensembels.append(tmp)
        ensemble_exits_logits = [pred_ensembels[i+1] for i in range(len(exits_logits))]
        return ensemble_exits_logits