import torch.cuda

import torch
import math
import torch.utils.checkpoint
import torch.nn.functional as F

from torch import nn
from typing import *
from einops import rearrange
from utils.train_utils import get_layer_idx


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def grads_to_named(self, layer_idx_range=None, include_IC=True)->Dict[str, torch.tensor]:
        named_grads = {}
        for idx, (name, param) in enumerate(self.named_parameters()):
            if layer_idx_range is not None:
                if len(layer_idx_range) == 1:
                    if get_layer_idx(name) != layer_idx_range: continue
                else:
                    if get_layer_idx(name) not in tuple(range(layer_idx_range)): continue
            if param.requires_grad is False: continue
            if 'classifier' in name & include_IC is False: continue
            named_grads[name] = param.grad.detach()
        return named_grads
            
    def parameters_to_tensor(self, blocks=(2,5,8,11), is_split=False, is_inclusivefl=False, is_scalefl=False, layers=None):
        if is_inclusivefl: blocks = (1,4,7,11)
        if is_scalefl: blocks = (3,6,9,11)
        if is_split:
            tensors = ()
            block_idx = 0
            params = []
            for idx, (name, param) in enumerate(self.named_parameters()):
                layer_idx = get_layer_idx(name)
                if layer_idx > blocks[block_idx]:
                    tensors += (torch.nan_to_num(torch.cat(params, 0), nan=0.0, posinf=0.0, neginf=0.0),)
                    block_idx += 1
                    params = []
                params.append(param.view(-1))
            if params != []: 
                tensors += (torch.nan_to_num(torch.cat(params, 0), nan=0.0, posinf=0.0, neginf=0.0),)
            return tensors
        else:
            params = []
            if layers is None:
                for idx, (name, param) in enumerate(self.named_parameters()):
                    params.append(param.view(-1))
                return torch.nan_to_num(torch.cat(params, 0), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                for idx, (name, param) in enumerate(self.named_parameters()):
                    if get_layer_idx(name) not in layers: continue
                    params.append(param.view(-1))
                return torch.nan_to_num(torch.cat(params, 0), nan=0.0, posinf=0.0, neginf=0.0)
        
        
    def split_state_dict(self, blocks=(2,5,8,11)):
        state_dict_tuple = ()
        block_idx = 0
        filter_state_dict = {}
        for idx, (name, param) in enumerate(self.named_parameters()):
            layer_idx = get_layer_idx(name)
            if layer_idx > blocks[block_idx]:
                state_dict_tuple += (filter_state_dict,)
                block_idx += 1
                filter_state_dict = {}
            filter_state_dict[name] = param
        if filter_state_dict != {}:
            state_dict_tuple += (filter_state_dict,)
        return state_dict_tuple
                

    def tensor_to_parameters(self, tensor, local_params=None, layers=None):
        param_index = 0
        if layers is None:
            for idx, (name, param) in enumerate(self.named_parameters()):
                # === get shape & total size ===
                shape = param.shape
                param_size = 1
                for s in shape:
                    param_size *= s

                # === put value into param ===
                # .clone() is a deep copy here
                param.data = tensor[param_index: param_index+param_size].view(shape).detach().clone()
                param_index += param_size
        else:
            for idx, (name, param) in enumerate(self.named_parameters()):
                if get_layer_idx(name) not in layers: continue
                shape = param.shape
                param_size = 1
                for s in shape:
                    param_size *= s
                param.data = tensor[param_index: param_index+param_size].view(shape).detach().clone()
                param_index += param_size
    
    
    def model_lora(self):
        params_with_grad = {name: param for name, param in self.named_parameters() if param.requires_grad}
        return params_with_grad
    
    def save_model(self, path):
        params_is_grads = self.model_lora()
        torch.save(params_is_grads, path)
        

class CNNCifar(BaseModule):
    def __init__(self, args, dim_out):
        super(CNNCifar, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)

        self.fc = nn.Linear(192, dim_out)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def features(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def logits(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

class MLP(BaseModule):
    def __init__(self, args, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.args = args
        self.layer_input = nn.Linear(dim_in, 512)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 64)

        self.fc = nn.Linear(64, dim_out)

    def features(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = F.relu(self.layer_input(x))
        x = F.relu(self.layer_hidden1(x))
        x = F.relu(self.layer_hidden2(x))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class CNNMnist(BaseModule):
    def __init__(self, args, dim_out):
        super(CNNMnist, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc = nn.Linear(64, dim_out)

    def features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(F.dropout2d(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# @dataclass
# class EncoderOutputRee(ModelOutput):
#     last_hidden_state: torch.FloatTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     ree_exit_outputs: Optional[Tuple[torch.FloatTensor]] = None




class NormAndLinear(nn.Module):
    def __init__(self, dim, num_classes, adapter=None, dropout=0., **kwargs):
        super().__init__()

        self.adapter = adapter

        self.layer_norm = nn.LayerNorm(dim)
        
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
    def forward(self, x):
        x = self.layer_norm(x)
        return self.mlp_head(x)

class Linear(nn.Module):
    def __init__(self, dim, num_classes, **kwargs):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)
                

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., attn_dim=0.):
        super().__init__()
        self.heads = heads
        if attn_dim == 0:
            attn_dim = dim

        self.scale = attn_dim ** -0.5

        self.to_qkv = nn.Linear(dim, attn_dim * 3, bias = True)
        self.to_out = nn.Sequential(
            nn.Linear(attn_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, recurrent_steps, heads, dropout, depth=1, attn_dim=0., mlp_ratio=4):
        super().__init__()
        self.recurrent_steps = recurrent_steps
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout, attn_dim=attn_dim))),
                Residual(PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for j in range(self.depth):
            for i in range(self.recurrent_steps):
                x = self.layers[j][0](x, mask = mask) # att
                x = self.layers[j][1](x) # ffn
        return x

class Ree(nn.Module):
    def __init__(self, 
        recurrent_steps, 
        heads, 
        depth, 
        base_model, 
        num_classes, 
        adapter=None, 
        dropout = 0., 
        emb_dropout = 0., 
        modulation=True, 
        exit_head='normlinear', # TODO: Pass in cls instead
        attn_dim=16, mlp_ratio=2,
        **kwargs):
        super().__init__()
        self.recurrent_steps = recurrent_steps 
        self.heads = heads 
        self.depth =  depth
        self.base_model = base_model
        self.num_classes = num_classes
        self.modulation = modulation # cls token modulation

        if 'base' in self.base_model:
            self.dim = 768
            self.pos_embedding = nn.Parameter(torch.zeros(1, 12+1, self.dim))
        elif 'small' in self.base_model:
            self.dim = 384
            self.pos_embedding = nn.Parameter(torch.zeros(1, 12+1, self.dim))
        elif 'XXS24' in self.base_model or 'vim' in self.base_model:
            self.dim = 192
            self.pos_embedding = nn.Parameter(torch.zeros(1, 24+1, self.dim))
        elif 'tiny' in self.base_model:
            self.dim = 192
            self.pos_embedding = nn.Parameter(torch.zeros(1, 12+1, self.dim))
        else: # resnet
            self.dim = 512
            self.pos_embedding = nn.Parameter(torch.zeros(1, 4+1, self.dim))
        
        self.client_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)

        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            # Cut & paste from PyTorch official master until it's in a few official releases - RW
            # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
            def norm_cdf(x):
                # Computes standard normal cumulative distribution function
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                            "The distribution of values may be incorrect.",
                            stacklevel=2)

            with torch.no_grad():
                # Values are generated by using a truncated uniform distribution and
                # then using the inverse CDF for the normal distribution.
                # Get upper and lower cdf values
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)

                # Uniformly fill tensor with values from [l, u], then translate to
                # [2l-1, 2u-1].
                tensor.uniform_(2 * l - 1, 2 * u - 1)

                # Use inverse cdf transform for normal distribution to get truncated
                # standard normal
                tensor.erfinv_()

                # Transform to proper mean, std
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)

                # Clamp to ensure it's in the proper range
                tensor.clamp_(min=a, max=b)
                return tensor
        
        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.client_token, std=.02)

        self.transformer = Transformer(self.dim, self.recurrent_steps, self.heads, dropout, depth=self.depth, attn_dim=attn_dim, mlp_ratio=mlp_ratio)

        exit_funcs = {'normlinear': NormAndLinear, 'linear': Linear}
        exit_func = exit_funcs[exit_head]

        self.head = exit_func(self.dim, self.num_classes, adapter=adapter)

    def forward(self, features, **kwargs):
        # features are cls_tokens
        b, n, _ = features.shape
        last_cls_token = features[:, -1]
        
        client_token = self.client_token.expand(b, -1, -1)
        x = torch.cat((client_token, features), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        m = self.transformer(x)

        return m

