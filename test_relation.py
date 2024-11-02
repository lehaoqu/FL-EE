import torch
import torch.nn as nn
from trainer.generator.generator import Generator_CIFAR, Generator_LATENT
import numpy as np
from PIL import Image
from utils.modelload.modelloader import load_model_eval
from utils.options import args_parser
import torch.nn.functional as F
import random
import math
from utils.train_utils import calc_target_probs
from dataset import (
    get_cifar_dataset,
    get_glue_dataset,
    get_svhn_dataset
)
from dataset.cifar100_dataset import CIFARClassificationDataset
from dataset.svhn_dataset import SVHNClassificationDataset
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, calc_target_probs, exit_policy

#  python test_relation.py depthfl boosted --dataset cifar100-224-d03 --model vit

def adapt_batch(data, args):
    batch = {}
    for key in data.keys():
        batch[key] = data[key].to(device)
        if key == 'pixel_values':
            if 'cifar' in args.dataset:
                batch[key] = CIFARClassificationDataset.transform_for_vit(batch[key])
            else:
                batch[key] = SVHNClassificationDataset.transform_for_vit(batch[key])
    label = batch['labels'].view(-1)
    return batch, label

def kd_criterion(pred, teacher):
    kld_loss = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=-1)
    softmax = nn.Softmax(dim=1)
    T=3
    _kld = kld_loss(log_softmax(pred/T), softmax(teacher/T)) * T * T
    return _kld

args = args_parser()
device = args.device
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank(2,3)

s = torch.tensor([[1,2],[3,4]])
t = torch.tensor([[2,3],[4,5]])
print(dist_criterion(s,t))

model_path = '/data/qvlehao/FL-EE/exps/cifar_0.05_tiny/largefl_cifar100-224-d03_vit_120c_1E_lrsgd0.05_base.pth'
config_path = '/data/qvlehao/FL-EE/exps/cifar_0.05_tiny/largefl_cifar100-224-d03_vit_120c_1E_lrsgd0.05_base.json'
global_model = load_model_eval(args, model_path, config_path).to(0)

ds = args.dataset
if 'cifar' in ds:
    get_dataset = get_cifar_dataset
elif 'svhn' in ds:
    get_dataset = get_svhn_dataset
else:
    ds = f'glue/{ds}'
    get_dataset = get_glue_dataset
    
train_dataset = get_dataset(args=args, path=f'dataset/{ds}/train/', eval_valids=True)
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, collate_fn=None)
print(len(train_dataset))



with torch.no_grad():
    f = []
    for i, data in enumerate(loader_train):
        batch, label = adapt_batch(data, args)
        exits_logits, exits_feature = global_model(**batch, is_latent=False, rt_feature=True)
        # t_feature = exits_feature[3]
        # norms = t_feature.norm(p=2, dim=1, keepdim=True)
        # t_feature = t_feature/norms
        # print(t_feature)
        # for exit_idx in range(4):
        #     exit_feature = exits_feature[exit_idx]
        #     exit_feature = exit_feature / exit_feature.norm(p=2, dim=1, keepdim=True)
        #     print('angle:', angle_criterion(exit_feature, t_feature))
        #     print('dist:', dist_criterion(exit_feature, t_feature))
        #     print('kd:', kd_criterion(exit_feature, t_feature))
        f.append(exits_feature[-1])
        first_logits = exits_logits[0]
        for exit_idx in range(4):
            exit_logits = exits_logits[exit_idx]
            print('angle:', angle_criterion(exit_logits, first_logits))
            print('dist:', dist_criterion(exit_logits, first_logits))
            print('kd:', kd_criterion(exit_logits, first_logits))
        if i == 2:
            break
    print('angle:', angle_criterion(f[0], f[1]))