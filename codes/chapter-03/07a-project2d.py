import numpy as np
import matplotlib.pyplot as plt

# 设置图片字体
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 16


def projection(u, v):
    """
    向量到向量的投影函数
    :param u: 向量 u
    :param v: 向量 v
    :return: 向量 u 在向量 v 上的投影
    """
    return (np.dot(u, v) / np.dot(v, v)) * v


def project_points(points, direction_vector, offset):
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



def plot_projection(points, projected_points, direction_vector, offset):
    """
    绘制原始点、投影点和直线。

    :param points: 原始点的二维数组 (n x 2)
    :param projected_points: 投影后的点的二维数组 (n x 2)
    :param direction_vector: 直线的方向向量
    :param offset: 直线的偏置 (截距)
    """
    # 绘制原始点和投影点之间的虚线
    for p, proj in zip(points, projected_points):
        plt.plot([p[0], proj[0]], [p[1], proj[1]], 'k--', alpha=0.3)

    # 绘制映射直线
    t = np.linspace(0, 6.5, 100)
    slop = direction_vector[1] / direction_vector[0]  # 映射直线的斜率
    line = np.array([t, slop * t]).T + offset         # 平移直线
    plt.plot(line[:, 0], line[:, 1], color='green', label='投影方向')

    # 绘制原始点
    plt.scatter(points[:, 0], points[:, 1], color='red', label='原始数据点', marker='^', s=100)

    # 绘制投影点
    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='blue', label='投影数据点', marker='o', s=100)



if __name__ == "__main__":
    plt.figure()

    # 示例数据和直线参数
    v = np.array([2, 1])  # 直线的方向向量
    b = 0                 # 偏置 (截距)

    points = np.array([
        [3, 4],
        [4, 1],
        [5, 3],
        [1, 4.5],
        [5.5, 0.5]
    ])

    # 计算投影后的点
    projected_points = project_points(points, v, np.array([0, b]))

    # 绘制结果
    plot_projection(points, projected_points, v, np.array([0, b]))


    plt.xlim(0, 7)
    plt.ylim(0, 5)
    # plt.axhline(0, color='black', linewidth=1)     # 水平线
    # plt.axvline(0, color='black', linewidth=1)     # 垂直线
    plt.gca().set_aspect('equal', adjustable='box')  # 坐标轴比例相同，确保投影看起来是垂直的
    plt.legend(prop={'family': 'SimSun'})

    # plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.07, wspace=0.15, hspace=0.1)

    plt.show()
