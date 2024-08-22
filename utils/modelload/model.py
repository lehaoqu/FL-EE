import torch.cuda

import torch
from torch import nn
import torch.nn.functional as F
from typing import *

import torch
import torch.utils.checkpoint
from torch import nn
from utils.train_utils import get_layer_idx

from transformers.utils import (
    logging,
)



# from models.utils.ree import Ree
logger = logging.get_logger(__name__)

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
            
    def parameters_to_tensor(self, blocks=(2,5,8,11), is_split=False, is_inclusivefl=False, is_scalefl=False):
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
            for idx, (name, param) in enumerate(self.named_parameters()):
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
                

    def tensor_to_parameters(self, tensor, local_params=None):
        param_index = 0
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


