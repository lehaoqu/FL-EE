import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTEmbeddings, ViTPreTrainedModel
from utils.modelload.modelloader import load_model
from utils.modelload.slimmable import set_width_ratio, convert_to_slimmable, custom_ops_dict, set_model_config
import numpy as np
import random

from thop import profile

seed = 1117
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    

class A: pass
args = A()
args.model = 'vit'
args.dataset = 'cifar100'
args.config_path = './models/facebook/deit-tiny-patch16-224'
args.policy = 'boosted'
args.alg = 'depthfl'
args.blocks = (2,5,8,11)
args.load_path = ''
args.ft = 'full'

ratios = [1.0, 0.75, 0.5, 0.25]

# dummy_input = {'pixel_values': torch.randn(1, 3, 224, 224).to(0)}

depth = 12


model = load_model(args, model_depth=12, is_scalefl=False, exits=(2,5,8,11))
# model.to(0)
# o1 = model(**dummy_input, stop_exit=depth)


# print(model)
model = convert_to_slimmable(model, ratios=ratios)
set_model_config(model.config.hidden_size, model.config.intermediate_size, model.config.num_attention_heads)
# model.to(0)
# o2 = model(**dummy_input, stop_exit=depth)

# print(model)
# assert torch.allclose(o1[3], o2[3], atol=1e-6), "Outputs do not match!"
# exit(0)


flops = {}


for ratio in ratios:
    set_width_ratio(ratio, model)
    print(model.config.hidden_size, model.config.intermediate_size)
    for depth in [3, 2, 1, 0]:
        from utils.train_utils import get_flops
        flops[(depth, ratio)] = get_flops(model, stop_exit=depth)
        # print(depth ,ratio, flops)

print(flops)
