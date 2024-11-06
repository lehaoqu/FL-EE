import torch
import torch.nn.functional as F
import torch.nn as nn

a1 = torch.rand(1,3, requires_grad=True)
a2 = torch.rand(3,4, requires_grad=True)
a = torch.matmul(a1,a2)

b1 = torch.rand(1,3, requires_grad=True)
b2 = torch.rand(3,4, requires_grad=True)
b = torch.matmul(a1,a2)

c = torch.zeros(2, 4)
c[0] = a
c[1] = b
d = torch.rand(2,4)

cir = nn.MSELoss()
l = cir(c,d)
l.backward()
print(a)
