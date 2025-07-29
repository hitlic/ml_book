import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

# 设置图片中文字体，确保能正确显示中文和数学符号
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 22
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'


def rbf_kernel(X1, X2, sigma=1.0):
    # ||x-y||^2 = x^2 + y^2 - 2xy
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    K = np.exp(-sigma * (X1_sq + X2_sq - 2 * X1 @ X2.T))
    return K


def poly_kernel(X1, X2, degree=3, coef0=1.0, gamma=1.0):
    # K(x, y) = (gamma * <x, y> + coef0) ** degree
    return (gamma * (X1 @ X2.T) + coef0) ** degree


def smo_kernel_svm(X, y, C=1.0, tol=1e-4, max_passes=20, kernel='rbf', kernel_args=None):
    N = X.shape[0]
    alphas = np.zeros(N)
    b = 0.0
    passes = 0

    # 1. 计算核矩阵
    if kernel == 'rbf':
        assert kernel_args is not None, "RBF kernel requires 'kernel_args' parameter."
        K = rbf_kernel(X, X, **kernel_args)
    elif kernel == 'poly':
        assert kernel_args is not None, "Polynomial kernel requires 'kernel_args' parameter."
        K = poly_kernel(X, X, **kernel_args)
    elif kernel == 'linear':
        K = X @ X.T
    else:
        raise ValueError('Unknown kernel')

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(N):
            # 2. 决策函数用核
            f_i = np.dot(alphas * y, K[:, i]) + b
            E_i = f_i - y[i]

            if (y[i] * E_i < -tol and alphas[i] < C) or (y[i] * E_i > tol and alphas[i] > 0):
                j = np.random.choice([k for k in range(N) if k != i])
                f_j = np.dot(alphas * y, K[:, j]) + b
                E_j = f_j - y[j]

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                if y[i] != y[j]:
                    L = max(0, alpha_j_old - alpha_i_old)
                    H = min(C, C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0, alpha_i_old + alpha_j_old - C)
                    H = min(C, alpha_i_old + alpha_j_old)

                if L == H:
                    continue

                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 1e-8:
                    continue

                alphas[j] = alpha_j_old + y[j] * (E_i - E_j) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    return alphas, b

def decision_function(X_train, y_train, alphas, b, X_test, kernel='rbf', kernel_args=None):
    sv = alphas > 1e-5
    X_sv = X_train[sv]
    y_sv = y_train[sv]
    a_sv = alphas[sv]
    if kernel == 'rbf':
        K = rbf_kernel(X_test, X_sv, **kernel_args)
    elif kernel == 'poly':
        K = poly_kernel(X_test, X_sv, **kernel_args)
    elif kernel == 'linear':
        K = X_test @ X_sv.T
    else:
        raise ValueError('Unknown kernel')
    return np.dot(K, a_sv * y_sv) + b


def classification_dataset():
    """构造数据集"""
    class1_1 = multivariate_normal([1, 1], 0.5 * np.eye(2), 25)
    class1_2 = multivariate_normal([5, 5], 0.5 * np.eye(2), 25)
    class2_1 = multivariate_normal([1, 5], 0.5 * np.eye(2), 25)
    class2_2 = multivariate_normal([5, 1], 0.5 * np.eye(2), 25)
    X = np.concatenate([class1_1, class1_2, class2_1, class2_2], axis=0)
    y = np.array([-1] * 50 + [1] * 50)
    return X, y

if __name__ == "__main__":
    np.random.seed(42)
    X, y = classification_dataset()

    C = 10

    # 核函数参数
    rbf_args = {'sigma': 0.2}
    poly_args = {'degree': 2, 'coef0': 1.0}

    # 使用RBF核进行SVM训练
    kernel = 'rbf'
    kernel_args = rbf_args

    # 使用多项式核进行SVM训练
    # kernel = 'poly'
    # kernel_args = poly_args


    alphas, b = smo_kernel_svm(X, y, C, max_passes=50, kernel=kernel, kernel_args=kernel_args)


    support_vectors_idx = np.where(alphas > 1e-5)[0]
    print(f"找到 {len(support_vectors_idx)} 个支持向量。")

    plt.figure(figsize=(10, 8))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', marker='+', s=150, label='正类')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', marker='_', s=150, label='负类')

    if len(support_vectors_idx) > 0:
        svs = X[support_vectors_idx]
        plt.scatter(svs[:, 0], svs[:, 1],
                    s=250, facecolors='none', edgecolors='k', linewidths=2, label='支持向量')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # 用核决策函数计算
    Z = decision_function(X, y, alphas, b, xy, kernel=kernel, kernel_args=kernel_args).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
               linestyles=['--', '-', '--'])
    ax.contourf(XX, YY, Z, levels=[-1, 1], alpha=0.1)

    # plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True, linestyle='--', alpha=0.6)
    if kernel == 'rbf':
        plt.text(1.8, -1.9, f'(a) rbf核 $\sigma={rbf_args["sigma"]}$')
    elif kernel == 'poly':
        plt.text(1.6, -1.9, f'(b) poly核 $d={poly_args["degree"]}$ $c={poly_args["coef0"]}$')
    plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.15, wspace=0.2, hspace=0.1)
    plt.show()
