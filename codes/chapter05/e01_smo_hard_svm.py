import numpy as np
import matplotlib.pyplot as plt

# 设置图片中文字体，确保能正确显示中文和数学符号
plt.rcParams['font.family'] = 'SimSun'  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.size'] = 22
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'


def smo_hard_svm(X, y, tol=1e-4, max_passes=20):
    N = X.shape[0]
    alphas = np.zeros(N)
    b = 0.0
    passes = 0

    # 预计算样本内积矩阵
    K = X @ X.T

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(N):
            # 计算预测值 f(x_i) 和误差 E_i
            f_i = np.dot(alphas * y, K[:, i]) + b
            E_i = f_i - y[i]

            # 检查样本i是否违反KKT条件 (y_i * f_i >= 1)
            # 1. 如果 alpha_i = 0, 那么 y_i * f_i 应该 >= 1。违反即 y_i * f_i < 1
            # 2. 如果 alpha_i > 0, 那么 y_i * f_i 应该 = 1。违反即 y_i * f_i != 1
            # 综合起来，寻找那些 y_i*E_i 不接近0且可以被优化的点
            if (y[i] * E_i < -tol and alphas[i] == 0) or (y[i] * E_i > tol and alphas[i] > 0) or (abs(y[i] * E_i) > tol and alphas[i] > 0):
                # 随机选择一个不等于i的j
                j = np.random.choice([k for k in range(N) if k != i])

                # 计算预测值 f(x_j) 和误差 E_j
                f_j = np.dot(alphas * y, K[:, j]) + b
                E_j = f_j - y[j]

                # 保存旧的alphas
                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                # 计算 alpha_j 的边界
                if y[i] != y[j]:
                    L = max(0, alpha_j_old - alpha_i_old)
                    H = float('inf')  # 无上界
                else:
                    L = 0
                    H = alpha_i_old + alpha_j_old

                if L == H:
                    continue

                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 1e-8:
                    continue

                # 更新 alpha_j 并进行裁剪
                alphas[j] = alpha_j_old + y[j] * (E_i - E_j) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                # 如果变化太小，则忽略
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                # 更新 alpha_i
                alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])

                # 更新 b
                b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if alphas[i] > 0:
                    b = b1
                elif alphas[j] > 0:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas += 1

        # 根据是否有alpha更新来调整passes计数
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

    return alphas, b


if __name__ == "__main__":
    # 随机生成二维线性可分数据
    np.random.seed(42)
    N = 50
    X_pos = np.random.randn(N, 2) + [2.5, 3]
    X_neg = np.random.randn(N, 2) + [-2.5, -3]
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(N), -np.ones(N)))

    # 训练模型
    alphas, b = smo_hard_svm(X, y, max_passes=50)

    # 计算权重 w
    w = np.sum(alphas[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)

    # 找到支持向量 (alpha > 0)
    support_vectors_idx = np.where(alphas > 1e-5)[0]
    print(f"找到 {len(support_vectors_idx)} 个支持向量。")
    if len(support_vectors_idx) > 0:
        print("支持向量的 Alphas:", alphas[support_vectors_idx])

    # --- 绘图 ---
    plt.figure(figsize=(10, 8))

    # 绘制所有样本点
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', marker='+', s=150, label='正类')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', marker='_', s=150, label='负类')

    # 显示支持向量
    if len(support_vectors_idx) > 0:
        plt.scatter(X[support_vectors_idx][:, 0], X[support_vectors_idx][:, 1],
                    s=200, facecolors='none', edgecolors='k', linewidths=2, label='支持向量')

    # 绘制决策边界和间隔
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = (xy @ w + b).reshape(XX.shape)

    # 绘制决策边界 (f(x)=0) 和间隔 (f(x)=+/-1)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])

    # 填充间隔区域
    ax.contourf(XX, YY, Z, levels=[-1, 1], alpha=0.1)

    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.09, wspace=0.2, hspace=0.1)
    plt.show()
