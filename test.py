import torch

from typing import *
from utils.train_utils import crop_tensor_dimensions
from dataset import get_glue_dataset, get_cifar_dataset


# def a_scale_tensors(tensors, samples):
    
#     def zero_pad(a, new_shape):
#         expanded_a = torch.zeros(new_shape, dtype=a.dtype)
#         start_indices = tuple(0 for _ in range(len(new_shape)))
#         end_indices = a.shape
#         index_tensor = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
#         expanded_a[index_tensor] = a
#         return expanded_a
            
    
#     weights = [torch.full(tensor.shape, sample) for (tensor, sample) in zip(tensors, samples)]
    
#     max_shape = tensors[-1].shape
    
#     global_tensor = torch.zeros(max_shape)
#     global_weight = torch.zeros(max_shape)
    
#     for idx, tensor in enumerate(tensors):
#         weighted_tensor = tensor * weights[idx]
#         weighted_tensor = zero_pad(weighted_tensor, max_shape)
#         global_tensor += weighted_tensor
        
#         weight = zero_pad(weights[idx], max_shape)
#         global_weight += weight
    
#     global_tensor = global_tensor / global_weight
#     return global_tensor

# a = torch.tensor([10,20,30])
# print(crop_tensor_dimensions(a, {3:2}))

# a = torch.tensor([[1,2, 4],
#                   [3,4, 5]])
# b = torch.tensor([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

# wa = 4
# wb = 8
# tensors = [a,b]
# samples = [wa,wb]
# print(a_scale_tensors(tensors, samples))

# a = torch.tensor([[[1,2,4],
#                    [5,4,9],
#                    [7,8,2],],
#                   [
#                    [5,6,1],
#                    [9,8,5],
#                    [4,5,5]
#                   ]])
# max_pres, argmax_pres = a.max(dim=2, keepdim=False)
# print(max_pres)
# print(argmax_pres)

# _,sorted_idx = max_pres.sort(dim=1, descending=True)
# print(_)
# print(sorted_idx)

# print(torch.range(1, 4))
# _p = torch.tensor([40 * (1.0/(40/2))], dtype=torch.float32)
# print(torch.log(_p))
# probs = torch.exp(torch.log(_p) * torch.tensor([i+1 for i in range(4)]))
# probs /= probs.sum()
# print(probs)
# dataset = get_glue_dataset(path=f'dataset/glue/sst2/train/0.npz', need_process=False)

dataset = get_cifar_dataset(path=f'dataset/cifar100-224-d03/valid/0.npz', need_process=False)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=None)

for i, data in enumerate(dataset_loader):
    print(data.keys())