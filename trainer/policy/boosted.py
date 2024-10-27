import torch

import torch.nn as nn
from typing import *

def add_args(parser):
    parser.add_argument('--ensemble_weight', type=float, default=0.2, help="ensemble weight")
    return parser

class Policy():
    def __init__(self, args):
        self.name = 'boosted'
        self.args = args
        self.device = self.args.device
        self.exits_num = self.args.exits_num
        self.loss_func = nn.CrossEntropyLoss()
        self.reweight = [self.args.ensemble_weight] * self.exits_num
    
    def train(self, model, batch, label, ws=None) -> torch.tensor:
        exits_logits = model(**batch)
        
        assert self.exits_num == len(exits_logits), f'expected {self.exits_num}, but {len(exits_logits)}'
        
        ws = [1 for i in range(self.exits_num)] if ws is None else ws
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i, logits in enumerate(exits_logits):
            tmp = (logits + pred_ensembels[-1]) * self.reweight[i]
            pred_ensembels.append(tmp)
            
        exits_loss = ()
        for i, logits in enumerate(exits_logits):
            pred_ensembel = pred_ensembels[i].detach()
            pred_final = pred_ensembel + logits
            exits_loss += (self.loss_func(pred_final, label) * ws[i],)
        return exits_loss, pred_ensembels[1:]
        

    def __call__(self, exits_logits):
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i, logits in enumerate(exits_logits):
            tmp = logits + pred_ensembels[-1] * self.reweight[i]
            pred_ensembels.append(tmp)
        ensemble_exits_logits = pred_ensembels[1:]
        return ensemble_exits_logits
    
    
    # == for finetune in server == 
    def sf(self, exits_logits):
        pred_ensembels = [torch.zeros(1).to(self.device)]
        for i, logits in enumerate(exits_logits):
            tmp = (logits + pred_ensembels[-1]) * self.reweight[i]
            pred_ensembels.append(tmp)

        ensembel_logits = []
        for i, logits in enumerate(exits_logits):
            pred_ensembel = pred_ensembels[i].detach()
            ensembel_logits.append(pred_ensembel + logits)

        return ensembel_logits
    
