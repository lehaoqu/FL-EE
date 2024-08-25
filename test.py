# import torch
# import torch.nn as nn
# import time

# class RandomLabelToHiddenStatus(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(RandomLabelToHiddenStatus, self).__init__()
#         # 定义网络层
#         self.fc1 = nn.Linear(input_dim, 512)  # 第一个全连接层
#         self.fc2 = nn.Linear(512, 512)       # 第二个全连接层
#         self.fc3 = nn.Linear(512, output_dim) # 第三个全连接层，输出维度匹配Transformer的隐藏状态

#     def forward(self, x):
#         # 定义前向传播
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)  # 不应用激活函数，因为Transformer的隐藏状态可以是任何值
#         return x

# # 输入维度是100，输出维度是197*386
# input_dim = 100
# output_dim = 197 * 386

# # 创建模型实例
# model = RandomLabelToHiddenStatus(input_dim, output_dim)

# # 创建一个随机标签的输入张量，大小为32x100
# random_labels = torch.randint(0, 100, (32, input_dim)).float()
# random_labels = random_labels.to(1)

# # 通过模型获取隐藏状态
# model.to(1)
# time.sleep(20)
# hidden_states = model(random_labels)

# # 调整隐藏状态的形状以匹配Transformer的期望输入，即32x197x386
# hidden_states = hidden_states.view(32, 197, 386)

# print(hidden_states.shape)  # 输出应该是torch.Size([32, 197, 386])

# import torch

# # 假设我们有一个包含形状为 1 的张量的元组
# tensors = (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]))
# weight = torch.tensor([2,3,4], dtype=torch.float)

# cat_tensor = torch.cat(tensors)
# s = torch.sum(weight*cat_tensor)

# print("Concatenated Tensor:\n", cat_tensor)
# print(s)

import torch.optim as optim
import torch

# 假设我们有一个模型
model = torch.nn.Linear(20, 50)

# 创建第一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 执行一些训练步骤...

# 现在创建第二个优化器，可能与第一个优化器有不同的参数
pseudo_optimizer = optim.SGD(model.parameters(), momentum=0.99)

# 将第一个优化器的状态复制到第二个优化器
pseudo_optimizer.load_state_dict(optimizer.state_dict())

# 此时，pseudo_optimizer 将具有与 optimizer 相同的学习率和其他设置
print("Learning rate:", pseudo_optimizer.param_groups[0]['lr'])