import torchvision

data_directory = 'ttt'
# transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()]
# )

# train_set = torchvision.datasets.SVHN(
#     root=data_directory, split="train", download=True, transform=transforms)
# test_set = torchvision.datasets.SVHN(
#     root=data_directory, split="test", download=True, transform=transforms)

import scipy.io as sio

# 读取.mat文件
mat_data = sio.loadmat('ttt/train_32x32.mat')
print(mat_data.keys())

# 提取变量
matrix1 = mat_data['X']
print(matrix1.shape)
matrix2 = mat_data['y']

# 显示变量信息
print("matrix1的形状:", matrix1.shape)
print("matrix2的数据类型:", type(matrix2))
