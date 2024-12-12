import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})

# 模拟数据
budgets = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]) * 10**8
resnets_acc = np.array([66, 68, 70, 72, 73, 74, 75])
densenets_acc = np.array([65, 67, 69, 71, 72, 73, 74])
sar_acc = np.array([67, 69, 71, 73, 74, 75, 76])
sss_acc = np.array([64, 66, 68, 70, 71, 72, 73])
tas_acc = np.array([65, 67, 69, 71, 72, 73, 74])
sdn_acc = np.array([63, 65, 67, 69, 70, 71, 72])
msdnet_acc = np.array([68, 70, 72, 73, 74, 75, 76])
ours_acc = np.array([70, 72, 74, 75, 75.5, 76, 76.5])

# 绘制图表
plt.figure(figsize=(8, 6))  # 设置图像大小为8x6英寸
plt.plot(budgets, resnets_acc, 'bs--', label='ResNets')
plt.plot(budgets, densenets_acc, 'yo--', label='DenseNets')
plt.plot(budgets, sar_acc, 'g^--', label='SAR')
plt.plot(budgets, sss_acc, 'mp--', label='SSS')
plt.plot(budgets, tas_acc, 'rd--', label='TAS')
plt.plot(budgets, sdn_acc, 'bv--', label='SDN')
plt.plot(budgets, msdnet_acc, 'k-', label='MSDNet')
plt.plot(budgets, ours_acc, 'r-', label='Ours')

# 添加注释
plt.annotate('+1.3 Acc', xy=(0.2*10**8, 68), xytext=(0.1*10**8, 67),
             arrowprops=dict(facecolor='cyan', shrink=0.05))
plt.annotate('×1.6 Speedup', xy=(0.5*10**8, 73), xytext=(0.4*10**8, 72),
             arrowprops=dict(facecolor='cyan', shrink=0.05))
plt.annotate('+1.1 Acc', xy=(0.8*10**8, 75), xytext=(0.7*10**8, 74),
             arrowprops=dict(facecolor='cyan', shrink=0.05))

# 设置图表标题和标签
plt.title('Budgeted batch classification on CIFAR-100', fontsize=14)
plt.xlabel('Average budget (in MUL-ADD) $\times 10^8$', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10, loc='best')  # 自动选择最佳位置

# 调整子图之间的间距
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

# 显示图表
plt.grid(True)
plt.show()
plt.savefig('./test.png')