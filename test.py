import torch
import torch.nn as nn

def kd_loss_func(pred, teacher, T=4.0):
    kld_loss = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=-1)
    softmax = nn.Softmax(dim=1)   # ← 注意这里 dim=1
    _kld = kld_loss(log_softmax(pred / T), softmax(teacher / T)) * T * T
    return _kld

# 测试：完全相同的 logits
x = torch.randn(32, 100)
loss = kd_loss_func(x, x, T=4.0)
print("KL loss:", loss.item())