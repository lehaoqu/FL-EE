
import numpy as np
import matplotlib.pyplot as plt

# 定义离散随机变量的值和对应的概率
values = np.array([1, 2, 3])
probabilities = np.array([0.2, 0.3, 0.5])

# 计算累积分布函数
cdf = np.cumsum(probabilities)

# 定义一个函数，用于生成连续随机变量的PDF
def generate_pdf(values, probabilities, num_points=1000):
    # 计算连续变量的值
    x = np.linspace(min(values) - 0.5, max(values) + 0.5, num_points)
    # 初始化PDF
    pdf = np.zeros_like(x)
    # 计算每个区间的PDF
    for i in range(len(values)-1):
        dx = values[i+1] - values[i]
        pdf[(x >= values[i]) & (x < values[i+1])] = probabilities[i] / dx
    # 处理最后一个区间
    if len(values) > 1:
        dx = x.max() - values[-1]
        pdf[(x >= values[-1])] = probabilities[-1] / dx if dx != 0 else 0
    else:
        # 如果只有一个值，PDF是一个在该值处的delta函数
        pdf[x == values[0]] = probabilities[0]
    return x, pdf

# 生成PDF
x, pdf = generate_pdf(values, probabilities)

# 绘制PDF
plt.figure(figsize=(8, 4))
plt.plot(x, pdf, label='PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Probability Density Function of Continuous Variable')
plt.legend()
plt.grid(True)
plt.savefig('t.png')
plt.show()

