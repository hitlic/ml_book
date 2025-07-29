import numpy as np
import matplotlib.pyplot as plt

# 设置图片中文字体，确保能正确显示中文和数学符号
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 22
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'


span = 3
n_samples = 20
sigma = 10.0 # RBF核的参数，控制核的宽度
lambda_reg = 0.1 # 正则化系数


# 1. 生成合成数据集
np.random.seed(42)

x_train = np.sort(np.random.rand(n_samples) * span).reshape(-1, 1) # 转为列向量
epsilon = np.random.normal(0, 0.1, n_samples) # 噪声
y_train = np.sin(2 * np.pi * x_train.flatten()) + 0.5 * x_train.flatten() + epsilon

# 真实函数用于绘图
x_true = np.linspace(0, span, 500).reshape(-1, 1) # 转为列向量
y_true = np.sin(2 * np.pi * x_true.flatten()) + 0.5 * x_true.flatten()


# 2. 定义高斯径向基核函数 (RBF Kernel)
def rbf_kernel(X1, X2, sigma):
    """
    利用高斯径向基核函数计算核矩阵
    X1: (N, D) 矩阵
    X2: (M, D) 矩阵
    返回: (N, M) 核矩阵
    """
    # 扩展维度以进行广播：
    # X1_expanded: (N, 1, D)
    # X2_expanded: (1, M, D)
    # 差值: (N, M, D)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    # 平方欧氏距离: (N, M)
    sq_dist = np.sum(diff**2, axis=2)
    return np.exp(-sigma * sq_dist)



# 3. 构建训练集的核矩阵 K
# K_train 的维度：(n_samples, n_samples)
K_train = rbf_kernel(x_train, x_train, sigma)

# 4. 求解系数 alpha
# alpha = (K + lambda * I)^-1 * y
I = np.eye(n_samples) # 单位矩阵
alpha = np.linalg.solve(K_train + lambda_reg * I, y_train)

# 5. 在测试集上进行预测
# K_test_pred 的维度：(len(x_true), n_samples)
K_test_pred = rbf_kernel(x_true, x_train, sigma)
# y_pred 的维度：(len(x_true),)
y_pred = K_test_pred @ alpha

# 6. 绘图
plt.figure(figsize=(10, 8))
plt.plot(x_true, y_true, label='真实函数: $\sin(2\pi x) + 0.5x$', color='blue', linestyle='--')
plt.scatter(x_train, y_train, label='训练样本点', color='red', marker='o', s=50, zorder=5)
plt.plot(x_true, y_pred, label='核岭回归预测', color='green')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-1.3, 3.5)
plt.legend(loc='upper left')
plt.grid(True)
plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.1, wspace=0.2, hspace=0.1)
plt.show()
