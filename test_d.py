import numpy as np

# 定义要生成的等分点数量
n = 5

# 使用numpy的linspace函数生成0到100的等分点
div_points = np.linspace(0, 100-1, n)

# 打印等分点
print("Division points:", div_points)