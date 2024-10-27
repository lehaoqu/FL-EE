import torch
import torch.nn as nn
import torch.optim as optim

# 假设我们有一些数据
# X是输入数据，y是目标输出
X = torch.randn(100, 1)  # 100个样本，每个样本10个特征
y = torch.randn(100, 1)   # 100个样本的目标输出

# 定义一个简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(10, 1)  # 输入特征数为10，输出特征数为1

    def forward(self, x):
        return self.linear(x)

# 实例化模型
model = LinearModel()

model.eval()
