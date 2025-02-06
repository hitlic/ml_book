import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import sqrt

# plt.rcParams['text.usetex'] = True  # 启用LaTeX渲染
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times']
# plt.rcParams['font.size'] = 16


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
fig = plt.figure(figsize=(6.5, 6))
ax2 = fig.add_subplot(111)

# 添加等高线的投影
ax2.contour(X, Y, Z, colors='black', linewidths=0.5, levels=3)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])


# 矩阵[[1, 0.5], [0.5, 1]]的特征向量矩阵
U = np.array([[1/sqrt(2), 1/sqrt(2)],
              [1/sqrt(2), - 1/sqrt(2)]])

# 原始坐标系中的横轴上的两点
x1 = np.array([[-2.2], [0]])
x2 = np.array([[2.2], [0]])

# 原始坐标系中的纵轴上的两点
y1 = np.array([[0], [-1.5]])
y2 = np.array([[0], [1.5]])

# 新坐标系横轴上的两点
x1_t, x2_t = U @ x1, U @ x2
print(x1_t, x2_t)
# 新坐标系纵轴上的两点
y1_t, y2_t = U @ y1, U @ y2
print(y1_t, y2_t)
# 绘制新坐标系的横轴
plt.arrow(x1_t[0][0], x1_t[1][0], x2_t[0][0] - x1_t[0][0], x2_t[1][0] - x1_t[1][0], head_width=0.08, head_length=0.12, fc='blue', ec='blue')
# 绘制新坐标系的纵轴
plt.arrow(y1_t[0][0], y1_t[1][0], y2_t[0][0] - y1_t[0][0], y2_t[1][0] - y1_t[1][0], head_width=0.08, head_length=0.12, fc='blue', ec='blue')

plt.text(x2_t[0][0]+0.1, x2_t[1][0]+0.1, '$x_1\'$')
plt.text(y2_t[0][0]+0.1, y2_t[1][0]-0.1, '$x_2\'$')

plt.tight_layout()
# plt.subplots_adjust(left=0.13, right=0.97, top=0.98, bottom=0.1, wspace=0.2, hspace=0.2)
plt.show()
