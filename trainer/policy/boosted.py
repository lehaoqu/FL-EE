import torch

import torch.nn as nn
from typing import *

def add_args(parser):
    return parser

class Policy():
    def __init__(self, args):
        self.name = 'boosted'
        self.args = args
        self.device = self.args.device
        self.exits_num = self.args.exits_num
        self.loss_func = nn.CrossEntropyLoss()
    
    def train(self, model, batch, label, ws=None) -> torch.tensor:
        exits_logits = model(**batch)
        
        assert self.exits_num == len(exits_logits), f'expected {self.exits_num}, but {len(exits_logits)}'
        
        ws = [1 for i in range(self.exits_num)] if ws is None else ws
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i, logits in enumerate(exits_logits):
            tmp = logits + pred_ensembels[-1]
            pred_ensembels.append(tmp)
            
        exits_loss = ()
        for i, logits in enumerate(exits_logits):
            pred_ensembel = pred_ensembels[i].detach()
            pred_final = pred_ensembel + logits
            exits_loss += (self.loss_func(pred_final, label) * ws[i],)
        return exits_loss, exits_logits
        

    def __call__(self, exits_logits):
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i, logits in enumerate(exits_logits):
            tmp = logits + pred_ensembels[-1]
            pred_ensembels.append(tmp)
        ensemble_exits_logits = [pred_ensembels[i+1] for i in range(len(exits_logits))]
        return ensemble_exits_logits
    
    # == for finetune in server == 
    def sf(self, exits_logits):
        if len(exits_logits) > 1:
            former_ensemble = sum(exits_logits[:-1])
            pred_final = exits_logits[-1] + former_ensemble.detach()
        else:
            pred_final = exits_logits[0]
        return pred_final