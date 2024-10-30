import torch
import torch.nn.functional as F

# 创建一个二维张量
tensor_2d = torch.tensor([[1, 1, 1], [1.0, 2.0, 3.0]])

# 对每一行进行单位化
normalized_tensor_2d = F.normalize(tensor_2d, p=2, dim=1)

print(normalized_tensor_2d)