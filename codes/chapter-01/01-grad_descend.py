import copy
import numpy as np
import random as rand
import matplotlib.pyplot as plt

DELTA = 0.001


def partial(f, x, d):
    """
    函数f对x的第d维的偏导
    :param f: 函数
    :param x: 求偏导的位置
    :param d: 求偏导的维度
    :return: 负偏导
    """
    _x = copy.copy(x)
    x_ = copy.copy(x)
    _x[d] -= DELTA
    x_[d] += DELTA
    return - (f(x_) - f(_x))/(2*DELTA)


def grad(f, x):
    """
    多元函数f在x处的梯度
    :param f: 函数
    :param x: 求梯度的位置，值为向量
    :return: 负梯度
    """
    return np.array([partial(f, x, i) for i, _ in enumerate(x)])


def fun(x):
    return x[0]**2 + 2*x[1] ** 2


def grad_desc(f, dim, init=None, learning_rate=0.1, max_step=20):
    if init is None:
        init = [rand.random() for _ in range(dim)]
    current_pos = np.array(init, dtype=float)
    track = []
    for _ in range(max_step):
        track.append(current_pos.tolist())
        g = grad(f, current_pos)
        current_pos += learning_rate * g
        if sum(np.abs(g)) < 1e-8:
            break
    track.append(current_pos)
    return np.stack(track)


if __name__ == '__main__':
    track = grad_desc(fun, 1, [-8, 8])

    print(track)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X = np.arange(-10, 10, 0.25)
    Y = np.arange(-10, 10, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = X**2 + 2*Y**2

    ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='hot', alpha=0.75)
    ax.plot(track[:, 0], track[:, 1], [fun(d) for d in track], c='b', marker='o')
    ax.view_init(51, -105)
    # plt.savefig('figure_a.png', dpi=400)
    fig = plt.figure()
    bx = fig.add_subplot(111)
    CS = bx.contour(X, Y, R)
    bx.clabel(CS, inline=1, fontsize=10)
    bx.plot(track[:, 0], track[:, 1], c='b', marker='o')
    # plt.savefig('figure_b.png', dpi=400)
    plt.show()
