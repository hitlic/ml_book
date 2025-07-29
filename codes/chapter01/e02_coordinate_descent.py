import numpy as np
import matplotlib.pyplot as plt

# 设置图片字体
# plt.rcParams['font.size'] = 13
# plt.rcParams['font.family'] = 'Times New Roman'


def coordinate_descent(partial_x, partial_y, initial_guess, step_size=0.1, tolerance=1e-6, max_iterations=1000):
    """
    坐标下降算法。
    Args:
        partial_x:      关于x的偏导函数
        partial_y:      关于y的偏导函数
        initial_guess:  初始值
        step_size:      步长
        tolerance:      用于判断是否收敛
        max_iterations: 最大迭代次数
    Returns:
        minimum_point:  最优值
        trajectory:     优化过程
    """
    x, y = initial_guess
    trajectory = [(x, y)]
    for _ in range(max_iterations):
        prev_x, prev_y = x, y
        # 固定y更新x
        x = prev_x - step_size * partial_x(prev_x, prev_y)
        trajectory.append((x, y))
        # 固定x更新y
        y = prev_y - step_size * partial_y(prev_x, prev_y)
        trajectory.append((x, y))
        # 判断是否收敛
        if abs(x - prev_x) < tolerance and abs(y - prev_y) < tolerance:
            break
    return (x, y), trajectory


def f(x, y):
    """优化目标函数"""
    return x**2 + 2*y**2


def partial_derivative_x(x, y):
    """目标函数对x的偏导"""
    return 2 * x


def partial_derivative_y(x, y):
    """目标函数对y的偏导"""
    return 4 * y


# 初始位置
initial_guess = (1.0, 1.0)

# 运行算法
minimum_point, trajectory = coordinate_descent(partial_derivative_x, partial_derivative_y, initial_guess)

# 可视化
trajectory = np.array(trajectory)
x_values = trajectory[:, 0]
y_values = trajectory[:, 1]

x = np.linspace(-0.2, 1.2, 100)
y = np.linspace(-0.2, 1.2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=50)
plt.scatter(x_values, y_values, color='red', label='优化过程')
plt.plot(x_values, y_values, color='red', linestyle='--')
plt.scatter(*minimum_point, color='green',  marker='*', s=200, label='最优值')

plt.legend(prop={'family': 'SimSun'}, loc='upper left')
# plt.colorbar()

# plt.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.07, wspace=0.2, hspace=0.1)

plt.show()
