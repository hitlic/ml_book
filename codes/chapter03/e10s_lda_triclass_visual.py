import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def projection(points_3d, v1, v2, p0):
    """
    将三维数据点投影到由基向量 v1 和 v2 确定的平面上，平面原点为 p0。

    :param points_3d (np.ndarray): 三维数据点，形状为 (n, 3)
    :param v1 (np.ndarray): 第一个基向量，形状为 (3,)
    :param v2 (np.ndarray): 第二个基向量，形状为 (3,)
    :param p0 (np.ndarray): 平面的原点，形状为 (3,)

    :Returns: np.ndarray: 投影后的点，形状为 (n, 3)。
    """
    # 确保基向量线性无关
    if np.linalg.matrix_rank(np.column_stack((v1, v2))) < 2:
        raise ValueError("基向量必须是线性无关的！")

    # 构建基矩阵 V
    V = np.column_stack((v1, v2))

    # 计算投影矩阵
    V_T = V.T
    gram_matrix = np.dot(V_T, V)
    gram_inv = np.linalg.inv(gram_matrix)
    projection_matrix = np.dot(V, np.dot(gram_inv, V_T))

    # 对所有点进行投影，投影公式: proj_V(u) = p0 + V (V^T V)^{-1} V^T (u - p0)
    points_proj = p0 + np.dot(projection_matrix, (points_3d - p0).T).T

    return points_proj


def plot_basis_plane(ax, v1, v2, p0, limitx=(0, 4), limity=(0, 4), alpha=0.5, color='g'):
    """
    绘制由基向量 v1 和 v2 确定的平面，平面原点为 p0。

    :param ax: matplotlib的 3D 坐标轴对象
    :param  (np.ndarray): 第一个基向量，形状为 (3,)
    :param v2 (np.ndarray): 第二个基向量，形状为 (3,)
    :param p0 (np.ndarray): 平面的原点，形状为 (3,)
    :param limitx (tuple): 平面x轴的绘制范围
    :param limity (tuple): 平面y轴的绘制范围
    :param alpha (float): 平面的透明度
    :param color (str): 平面的颜色
    """
    # 平面的法向量
    normal_vector = np.cross(v1, v2)

    # 生成平面网格
    xx, yy = np.meshgrid(np.linspace(limitx[0], limitx[1], 10), np.linspace(limity[0], limity[1], 10))
    zz = (-normal_vector[0] * (xx - p0[0]) - normal_vector[1] * (yy - p0[1])) / normal_vector[2] + p0[2]

    # 绘制平面
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color, label='最优投影平面')
    ax.plot_wireframe(xx, yy, zz, color='brown', linewidth=0.5, alpha=0.3, rstride=1, cstride=1)


def plot_projected_points(ax, points_3d, points_proj, labels, color_original, color_projected, markers, line_color='k'):
    """
    绘制原始点和投影点，并绘制它们之间的连线。

    参数:
        ax (Axes3D): matplotlib 的 3D 坐标轴对象。
        points_3d (np.ndarray): 原始三维数据点，形状为 (n, 3)。
        points_proj (np.ndarray): 投影后的点，形状为 (n, 3)。
        color_original (str): 原始点的颜色。
        color_projected (str): 投影点的颜色。
        line_color (str): 连线的颜色。
    """
    # 绘制原始点
    for i, lable in enumerate(np.unique(labels)):
        class_datas = points_3d[labels == lable]
        ax.scatter(class_datas[:, 0], class_datas[:, 1], class_datas[:, 2],
               color=color_original[i], label=f'类别{i}原始数据', s=80, marker=markers[i])

        # 绘制投影点
        class_datas = points_proj[labels == lable]
        ax.scatter(class_datas[:, 0], class_datas[:, 1], class_datas[:, 2],
               color='w', edgecolors=color_projected[i], label=f'类别{i}投影数据', s=60, marker=markers[i])

    # 绘制从原点到投影点的线
    for i in range(len(points_3d)):
        ax.plot([points_3d[i, 0], points_proj[i, 0]],
                [points_3d[i, 1], points_proj[i, 1]],
                [points_3d[i, 2], points_proj[i, 2]],
                color=line_color, linestyle='--', alpha=0.3
                )


if __name__ == "__main__":

    # 设置随机种子
    np.random.seed(1234)

    # 生成三维数据集
    points_3d, label = make_blobs(
        n_samples=100,
        n_features=3,           # 三维数据
        centers=3,              # 3个类别
        cluster_std=1.0,        # 每个类别的高斯分布标准差
        center_box=(-5.0, 5.0),  # 每个类别中心的边界范围
        random_state=42         # 随机种子
    )

    # 基向量
    v1 = np.array([0.49340483, -0.81213164, -0.31143839])
    v2 = np.array([-0.87002459, -0.16411093, -0.46489225])

    # 平面平移向量
    p0 = np.array([0, 0, -8])  # 平面的原点，用于平移平面位置

    # 平面绘制范围
    limitx = (-5, 6)
    limity = (-4.5, 11)

    # 投影数据点
    points_proj = projection(points_3d, v1, v2, p0)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制平面
    plot_basis_plane(ax, v1, v2, p0, limitx, limity, alpha=0.3, color='yellow')

    # 绘制原始点和投影点
    plot_projected_points(ax, points_3d, points_proj, label,
                          color_original=['r', 'g', 'b'], color_projected=['r', 'g', 'b'], markers=['^', 'o', 's'])

    # # 设置坐标轴范围
    ax.set_xlim([-8, 8])
    ax.set_ylim([-5, 12])
    ax.set_zlim([-11, 4])

    ax.legend(prop={'family': 'SimSun', 'size': 14})
    plt.gca().set_aspect('equal', adjustable='box')  # 坐标轴比例相同

    plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.0, wspace=0.15, hspace=0.1)

    plt.show()
