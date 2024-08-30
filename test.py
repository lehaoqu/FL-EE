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

# import torch.optim as optim
# import torch

# # 假设我们有一个模型
# model = torch.nn.Linear(20, 50)

# # 创建第一个优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # 执行一些训练步骤...

# # 现在创建第二个优化器，可能与第一个优化器有不同的参数
# pseudo_optimizer = optim.SGD(model.parameters(), momentum=0.99)

# # 将第一个优化器的状态复制到第二个优化器
# pseudo_optimizer.load_state_dict(optimizer.state_dict())

# # 此时，pseudo_optimizer 将具有与 optimizer 相同的学习率和其他设置
# print("Learning rate:", pseudo_optimizer.param_groups[0]['lr'])

import torch
import torch.nn as nn
import torch.nn.functional as F

# # 定义一个简单的模型
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.fc2 = nn.Linear(5, 2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 创建模型实例
# model = SimpleModel()

# # 创建数据和目标
# x = torch.randn(1, 10, requires_grad=True)
# y = torch.tensor([1], dtype=torch.long)

# # 前向传播
# output = model(x)

# # 计算损失
# criterion = nn.CrossEntropyLoss()
# loss = criterion(output, y)

# # 第一次反向传播，保留计算图
# loss.backward(retain_graph=True)

# # 打印第一次梯度
# print("First-order gradients:")
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"{name}: {param.grad}")

# # 假设我们需要计算Hessian矩阵的近似，进行第二次反向传播
# # 这里只是一个示例，实际上计算Hessian需要更复杂的步骤
# hessian_vector_product = torch.autograd.grad(
#     outputs=output,
#     inputs=x,
#     grad_outputs=torch.ones_like(output),
#     create_graph=True,
#     retain_graph=True
# )[0]

# # 打印Hessian向量乘积的结果
# print("\nHessian-vector product:")
# print(hessian_vector_product)

# # 清理不再需要的计算图
# del hessian_vector_product

# # 进行第二次反向传播，如果需要
# loss.backward()

# # 打印第二次梯度
# print("Second-order gradients:")
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"{name}: {param.grad}")

# class GradientRescaleFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.save_for_backward(input)
#         ctx.gd_scale_weight = weight
#         output = input
#         return output
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input = ctx.saved_tensors
#         grad_input = grad_weight = None
#         if ctx.needs_input_grad[0]:
#             grad_input = ctx.gd_scale_weight * grad_output
#         return grad_input, grad_weight


# gradient_rescale = GradientRescaleFunction.apply

# x = torch.tensor([1,2,3], dtype=torch.float, requires_grad=True)
# z = x*x
# # z = gradient_rescale(z, 0.5)
# l = torch.sum(z)
# l.backward()
# print(x.grad)

import torch
import time

# # 一个包含PyTorch张量的元组
# tensors = (torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(4))

# # 求和
# total = sum(tensors)  # 直接使用sum，因为PyTorch重载了sum函数

# print(total)  # 输出: tensor(10)

# class A(nn.Module):
#     def __init__(self,):
#         super(A, self).__init__()
#         self.l1 = nn.Linear(100, 384*128)

    
# a = A()
# a.to(0)
# time.sleep(100)
# t = torch.tensor([1, 2])
# q = torch.tensor([3, 4])
# a = (t, q)
# print(sum(a))

# hidden_states = torch.rand(12,13,14)
# a = torch.rand(12,13,14)
# print(hidden_states)
# a[:, 0] = hidden_states[:, 0]
# print(a)

class A(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.liner = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.liner(x)

x = torch.tensor([1.,2.,3.,4.,5.], dtype=torch.float32)
a = A()

y = a(x)
y.backward()
print(x.grad)