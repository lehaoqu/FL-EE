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
from utils.train_utils import RkdDistance, RKdAngle, HardDarkRank, calc_target_probs, exit_policy, difficulty_measure

#  python load_generator.py depthfl boosted --dataset cifar100-224-d03 --model vit

def get_batch(args, gen_latent, y_input):
    batch = {}
    if 'cifar' in args.dataset:
        batch['pixel_values'] = gen_latent
    else:
        batch['input_ids'] = gen_latent
        batch['attention_mask'] = y_input[1]
    return batch

# a = torch.load('./_3.pth')
# g = Generator_CIFAR()
# g.load_state_dict(a)
# g.to(0)
# y_input = torch.tensor([52,52,52,52,52], dtype=torch.long).to(0)
# diffs = torch.tensor([1,3,5,7,9], dtype=torch.long).to(0)
# eps = torch.rand((y_input.shape[0], g.noise_dim)).to(0)

# with torch.no_grad():
#     imgs = g(diffs, y_input, eps, raw=True)
#     for i, img in enumerate(imgs):
#         array = np.transpose(img.cpu().numpy(), (1, 2, 0))
#         img = Image.fromarray(array.astype(np.uint8))
#         img.save(f'generators/dlevel_{diffs[i].cpu().item()}_l_{y_input[i].cpu().item()}.png')


args = args_parser()
args.diff_generator=True
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

a = torch.load('exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted_G_12.pth')
g = Generator_LATENT(args)

g.load_state_dict(a, strict=False)

g.to(0)
y_input = torch.tensor([30]*5, dtype=torch.long).to(0)
diffs = torch.tensor([0,0,0,0,0]*1, dtype=torch.float).resize(5, 1).to(0)
eps = torch.rand((y_input.shape[0], g.noise_dim)).to(0)

model_path = 'exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.pth'
config_path = 'exps/BASE_CIFAR/full_boosted/noniid1000/darkflpg_cifar100_noniid1000_vit_100c_1E_lrsgd0.05_boosted.json'
global_model = load_model_eval(args, model_path, config_path).to(0)
exits_num = 4
target_probs = calc_target_probs(exits_num)[15-1]
with torch.no_grad():
    imgs = g(y_input, eps, diffs)
    exits_logits = global_model(**get_batch(args, imgs, y_input), is_latent=True, rt_feature=False)
    print(exits_logits)
    
    used_index, ce_loss = [], 0.0
    
    for j in range(exits_num):
        print(f'exit {j}')
        with torch.no_grad():
            confidence_target = F.softmax(exits_logits[j], dim=1)  
            # print(confidence_target)
            # exit(0)
            max_preds_target, _ = confidence_target.max(dim=1, keepdim=False)  
            _, sorted_idx = max_preds_target.sort(dim=0, descending=True)  
            n_target = sorted_idx.shape[0]
            
            if j == 0:
                selected_index = sorted_idx[: math.floor(n_target * target_probs[j])]
                selected_index = selected_index.tolist()
                used_index.extend(selected_index)
            elif j < exits_num - 1:
                filter_set = set(used_index)
                unused_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                selected_index = unused_index[: math.floor(n_target * target_probs[j])]  
                used_index.extend(selected_index)
            else:
                filter_set = set(used_index)
                selected_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
        
        if len(selected_index) > 0:
            sum_d = 0.0
            for sample_index in selected_index:
                last_logits = exits_logits[-1][sample_index].unsqueeze(0)
                diff_pred = 0
                # for exit_idx in range(len(exits_logits)):
                # exit_logits = exits_logits[0][sample_index].unsqueeze(0)
                    
                diff_pred = difficulty_measure([exits_logits[0][sample_index]], y_input[sample_index], metric='loss', rt_exits_diff=False)
                    # diff_pred += nn.functional.cosine_similarity(exit_logits, last_logits, dim=1)
                d = diff_pred
                sum_d += d.cpu().item()    
            print(sum_d/len(selected_index))
    
