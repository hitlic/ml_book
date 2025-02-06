import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 定义二维正态分布的参数
mean = [0, 0]
covariance = [[1, 0.5], [0.5, 1]]

# 创建一个网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 计算二维正态分布的概率密度函数值
rv = multivariate_normal(mean, covariance)
Z = rv.pdf(pos)

# 创建图像
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')

# 绘制三维概率密度图像
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.view_init(elev=45, azim=105)

# 添加等高线的投影
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, colors='black', linewidths=0.5)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])

plt.tight_layout()
# plt.savefig('fig.png', dpi=300)
plt.show()
