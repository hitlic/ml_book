import numpy as np
import matplotlib.pyplot as plt


# 设置图片中文字体，确保能正确显示中文和数学符号
plt.rcParams['font.family'] = 'SimSun'  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'



def z_fn(x, y, center_x=3, center_y=3):
    """特征映射函数，计算点到中心的距离平方"""
    return (x - center_x)**2 + (y - center_y)**2

# --- 1. 准备数据 ---
# 数据集A (靠近中心)
data1 = np.array([[2, 3], [3, 2], [3, 4], [4, 3]])
x1, y1 = data1[:, 0], data1[:, 1]
z1 = z_fn(x1, y1)

# 数据集B (远离中心)
data2 = np.array([[1, 3], [2, 1.5], [2, 4.5], [4, 1.5], [4, 4.5], [5, 3]])
x2, y2 = data2[:, 0], data2[:, 1]
z2 = z_fn(x2, y2)

# --- 2. 创建图形和子图 ---
fig = plt.figure(figsize=(16, 7))

# --- 3. 左侧子图: 原始二维空间 ---
ax1 = fig.add_subplot(121)
ax1.scatter(x1, y1, c='#d62728', label='类别 A', s=80, marker='o', edgecolors='k')
ax1.scatter(x2, y2, c='#2ca02c', label='类别 B', s=80, marker='^', edgecolors='k')

# 绘制一个理想的决策边界（圆）来强调不可分性
circle = plt.Circle((3, 3), 1.3, color='blue', fill=False, linestyle='--', linewidth=2, label='理想决策边界')
ax1.add_artist(circle)

ax1.set_title('（a）原始二维空间 (线性不可分)', y=-0.17)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 6)
ax1.set_aspect('equal', adjustable='box') # 保持x,y轴比例相同
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left')


# --- 4. 右侧子图: 映射后的三维空间 ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x1, y1, z1, c='#d62728', label='类别 A', s=80, marker='o', depthshade=False)
ax2.scatter(x2, y2, z2, c='#2ca02c', label='类别 B', s=80, marker='^', depthshade=False)

# 绘制一个分离平面 z = 3
xx, yy = np.meshgrid(np.linspace(0, 6, 10), np.linspace(0, 6, 10))
zz = np.full_like(xx, 2) # 平面的z值
ax2.plot_surface(xx, yy, zz, alpha=0.3, color='blue', rstride=100, cstride=100, label='分离超平面')

ax2.set_title('（b）映射到三维空间 (线性可分)', y=-0.17)
ax2.set_xlabel('$x_1$')
ax2.set_ylabel('$x_2$')
ax2.set_zlabel('$x_3$')
ax2.text(4, 0, -0.7, '$x_3 = (x_1-3)^2 + (x_2-3)^2$')
ax2.legend(loc='upper left')
ax2.view_init(elev=12, azim=-50) # 调整视角以便观察
ax2.grid(True)

# --- 5. 显示图像 ---
# plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(left=0.0, right=0.99, top=0.97, bottom=0.15, wspace=0.2, hspace=0.1)
plt.show()
