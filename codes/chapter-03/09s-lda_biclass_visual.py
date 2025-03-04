import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


# 设置随机种子
np.random.seed(12)

# 向量投影函数
def projection(u, v):
    return (np.dot(u, v) / np.dot(v, v)) * v

# 投影函数
def project_points_on_line(points, direction_vector, offset):
    projected_points = []
    for p in points:
        shifted_point = p - offset
        proj_point = projection(shifted_point, direction_vector)
        projected_points.append(proj_point + offset)
    return np.array(projected_points)


# 绘制投影直线
def plot_projection_line(direction_vector, offset, lower=0, upper=6.5, color='g', label=None):
    t = np.linspace(lower, upper, 100)
    slop = direction_vector[1] / direction_vector[0]
    line = np.array([t, slop * t]).T + offset
    plt.plot(line[:, 0], line[:, 1], color=color, label=label, linewidth=2)


# 绘制数据点
def plot_data(points, labels, colors, markers):
    unique_labels = np.unique(labels)
    assert len(unique_labels) == len(colors) and len(unique_labels) == len(markers)
    for c_id, label in enumerate(unique_labels):
        c_points = points[labels == label]
        plt.scatter(c_points[:, 0], c_points[:, 1], color=colors[c_id], marker=markers[c_id], s=80, label=f'类别 {label}', edgecolors='k')


# 绘制投影后的数据点
def plot_projected_data(points, projected_points, labels, colors, markers):
    unique_labels = np.unique(labels)
    assert len(unique_labels) == len(colors) and len(unique_labels) == len(markers)
    for c_id, label in enumerate(unique_labels):
        c_points = points[labels == label]
        c_projected_points = projected_points[labels == label]
        for p, proj in zip(c_points, c_projected_points):
            plt.plot([p[0], proj[0]], [p[1], proj[1]], 'k--', alpha=0.2)
        plt.scatter(c_projected_points[:, 0], c_projected_points[:, 1], color='w', marker=markers[c_id], edgecolors=colors[c_id], alpha=0.6, s=80)


if __name__ == "__main__":

    # 生成数据集
    data, label = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,
        cluster_std=1.0,
        center_box=(-5.0, 5.0),
        random_state=42
    )

    # 创建图形
    # plt.figure(figsize=(8, 6))

    # 绘制原始数据
    colors = ['r', 'b']
    markers = ['o', 's']
    plot_data(data, label, colors, markers)

    # 定义投影直线的方向向量和偏置
    v = np.array([0.83407968, -0.55164399])
    b = -4

    # 投影数据
    projected_data = project_points_on_line(data, v, np.array([0, b]))

    # 绘制投影直线
    plot_projection_line(v, np.array([0, b]), lower=-8, upper=2.5, color='g', label="最优投影方向")

    # 绘制投影后的数据
    plot_projected_data(data, projected_data, label, colors, markers)


    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', prop={'family': 'SimSun'})
    plt.xlim(-10.5, 8.5)
    plt.ylim(-6.5, 9.9)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, wspace=0.15, hspace=0.1)
    plt.show()
