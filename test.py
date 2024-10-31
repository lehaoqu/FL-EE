import torch
import torch.nn.functional as F

c = torch.tensor([3, 4])
b = torch.tensor([4, 5])

t = F.pairwise_distance(b,c)
print(t)