import torch
import torch.nn as nn
import torch.optim as optim

# 假设我们有一些数据
# X是输入数据，y是目标输出
X = torch.randn(100, 10)  # 100个样本，每个样本10个特征
y = torch.randint(0, 2, (100, 1))  # 100个样本的目标输出，随机生成0或1

# 将y转换为float类型
y = y.float()
# 定义一个简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(10, 1)  # 输入特征数为10，输出特征数为1

    def forward(self, x):
        return self.linear(x)

# 实例化模型
model = LinearModel()

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降，学习率为0.01

# 训练模型
num_epochs = 100  # 训练的轮数
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 打印损失值
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 训练完成后，模型的参数已经更新