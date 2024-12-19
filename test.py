import numpy as np
import matplotlib.pyplot as plt


# 假设这些是你的数据点坐标（示例数据）
x1 = np.random.randn(100)
y1 = np.random.randn(100)
x2 = np.random.randn(100)
y2 = np.random.randn(100)
x3 = np.random.randn(100)
y3 = np.random.randn(100)
x4 = np.random.randn(100)
y4 = np.random.randn(100)

# 绘制散点图
plt.scatter(x1, y1, c='red', label='Difficulty 1', s=50, marker='s')  # 方形，大小为50
plt.scatter(x2, y2, c='purple', label='Difficulty 2', s=30, marker='^')  # 三角形，大小为30
plt.scatter(x3, y3, c='green', label='Difficulty 3', s=40, marker='o')  # 圆形，大小为40
plt.scatter(x4, y4, c='brown', label='Difficulty 4', s=60, marker='*')  # 星形，大小为60

# 设置标题
plt.title("Pseudo latent Visualization Label: 0")

# 添加图例
plt.legend()

# 显示图形
plt.show()
plt.savefig('t.png')