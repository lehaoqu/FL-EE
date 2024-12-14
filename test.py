from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 假设X是包含多个向量的数组，shape为(num_samples, num_features)
X = np.random.rand(32, 4)*0.1  # 示例数据




# 假设data是一个包含多个向量的NumPy数组，每个向量有10个维度


# 创建t-SNE对象，并指定降维后的维度为2
tsne = TSNE(n_components=2)

# 对数据进行降维
result = tsne.fit_transform(X)

# 可视化降维后的结果
plt.scatter(result[:, 0], result[:, 1])



X = np.random.rand(32, 4)*10
tsne = TSNE(n_components=2)

# 对数据进行降维
result = tsne.fit_transform(X)

# 可视化降维后的结果
plt.scatter(result[:, 0], result[:, 1], color='red')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Vectors')
plt.show()
plt.savefig('test.png')