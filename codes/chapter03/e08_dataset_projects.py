import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 设置图片字体
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 14


np.random.seed(12)



def projection(u, v):
    """
    向量到向量的投影函数
    :param u: 向量 u
    :param v: 向量 v
    :return: 向量 u 在向量 v 上的投影
    """
    return (np.dot(u, v) / np.dot(v, v)) * v


def project_points_on_line(points, direction_vector, offset):
    """
    计算一组点在指定直线上的投影。

    :param points: 原始点的二维数组 (N x 2)
    :param direction_vector: 直线的方向向量
    :param offset: 直线的偏置 (截距)
    :return: 投影后的点的二维数组 (N x 2)
    """
    projected_points = []

    for p in points:
        shifted_point = p - offset  # 根据指定偏置平移直线
        proj_point = projection(shifted_point, direction_vector)  # 计算平移后的投影
        projected_points.append(proj_point + offset)  # 将投影点反向平移回去

    return np.array(projected_points)


def plot_projection_line(direction_vector, offset,lower=0, upper=6.5,  color='g', label=None):
    """绘制映射直线"""
    t = np.linspace(lower, upper, 100)
    slop = direction_vector[1] / direction_vector[0]  # 映射直线的斜率
    line = np.array([t, slop * t]).T + offset         # 平移直线
    plt.plot(line[:, 0], line[:, 1], color=color, label=label)


def plot_data(points, labels, colors, markers):
    unique_labels = np.unique(labels)
    assert len(unique_labels) == len(colors) and len(unique_labels) == len(markers)
    for c_id, label in enumerate(unique_labels):
        c_points = points[labels == label]
        plt.scatter(c_points[:, 0], c_points[:, 1], color=colors[c_id], marker=markers[c_id], edgecolors='k', s=70)


def plot_projected_data(points, projected_points, labels, colors, markers):
    unique_labels = np.unique(labels)
    assert len(unique_labels) == len(colors) and len(unique_labels) == len(markers)
    for c_id, label in enumerate(unique_labels):
        # 获取某一类样本的原始位置和映射位置
        c_points = points[labels == label]
        c_projected_points = projected_points[labels == label]

        # 绘制原始点和投影点之间的虚线
        for p, proj in zip(c_points, c_projected_points):
            plt.plot([p[0], proj[0]], [p[1], proj[1]], 'k--', alpha=0.1)

        # 绘制投影点
        plt.scatter(c_projected_points[:, 0], c_projected_points[:, 1], color='w', marker=markers[c_id], edgecolors=colors[c_id], alpha=0.4)


if __name__ == "__main__":

    # 生成数据集
    data, label = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,               # 类别数量
        cluster_std=1.0,         # 每个类别的高斯分布标准差
        center_box=(-5.0, 5.0),  # 每个类别中心的边界范围
        random_state=42          # 随机种子
    )


    plt.figure()

    # --- 绘制原始数据点
    colors = ['r', 'b']
    markers = ['o', 's']
    plot_data(data, label, colors, markers)


    # --- 第一种投影
    colors = ['r', 'b']
    markers = ['o', 's']

    v = np.array([2, 1])   # 投影向量
    b = -5                 # 直线偏移量

    # 投影
    projected_data = project_points_on_line(data, v, np.array([0, b]))
    plot_projection_line(v, np.array([0, b]), lower=0, upper=7)
    plot_projected_data(data, projected_data, label, colors, markers)


    # --- 第二种投影
    colors = ['r', 'b']
    markers = ['o', 's']

    v = np.array([-2, 4])   # 投影向量
    b = 20                  # 直线偏移量

    # 投影
    projected_data = project_points_on_line(data, v, np.array([0, b]))
    plot_projection_line(v, np.array([0, b]), lower=4.5, upper=9.5)
    plot_projected_data(data, projected_data, label, colors, markers)


    plt.gca().set_aspect('equal', adjustable='box')  # 坐标轴比例相同，确保投影看起来是垂直的

    plt.xlim(-7.5, 12)
    plt.ylim(-5.5, 12)

    # plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.07, wspace=0.15, hspace=0.1)

    plt.show()
