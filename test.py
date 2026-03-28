import torch
import torch.nn as nn

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]])

l = nn.MSELoss()(a,b)
print(a)
print(b)
print('-----')
print(l)