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
n_samples = 20   # 训练样本数量
m_anchors = 10   # 锚点数量
sigma = 10.0     # RBF核的参数，控制核的宽度

fig_title = f'(b) 锚点数量 $m={m_anchors}$'

# 1. 生成合成数据集
np.random.seed(42)

x_train = np.sort(np.random.rand(n_samples) * span).reshape(-1, 1)
epsilon = np.random.normal(0, 0.1, n_samples) # 噪声

y_train = np.sin(2 * np.pi * x_train.flatten()) + 0.5 * x_train.flatten() + epsilon

x_true = np.linspace(0, span, 500).reshape(-1, 1)
y_true = np.sin(2 * np.pi * x_true.flatten()) + 0.5 * x_true.flatten()

# 2. 利用高斯径向基核函数计算核矩阵
def rbf_kernel(X1, X2, sigma):
    """
    高斯径向基核函数 - 处理矩阵输入
    X1: (N, D) 矩阵
    X2: (M, D) 矩阵
    返回: (N, M) 核矩阵
    """
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    return np.exp(-sigma * sq_dist)



# 3. 随机采样锚点
# 从训练数据中随机选择 m_anchors 个点作为锚点
anchor_indices = np.random.choice(n_samples, m_anchors, replace=False)
anchor_points = x_train[anchor_indices]

y_anchors = y_train[anchor_indices]

# 4. 构建核特征矩阵 Phi，维度： (n_samples, m_anchors)
Phi_train = rbf_kernel(x_train, anchor_points, sigma)

# 5. 求解权重向量 w (假设矩阵可逆)
try:
    w = np.linalg.solve(Phi_train.T @ Phi_train, Phi_train.T @ y_train)
except np.linalg.LinAlgError as e:
    raise ValueError("警告: 矩阵 (Phi_train.T @ Phi_train) 不可逆，无法使用基本线性回归的正规方程。") from e


# 6. 在测试集上进行预测
# Phi_test 的维度：(len(x_true), m_anchors)
Phi_test = rbf_kernel(x_true, anchor_points, sigma)

# 预测值
y_pred = Phi_test @ w

# 7. 绘图
plt.figure(figsize=(10, 8))
plt.plot(x_true, y_true, color='blue', linestyle='--')  # label='$\sin(2\pi x) + 0.5x$',
plt.scatter(x_train, y_train, label='训练样本', color='red', marker='o', s=50)
plt.scatter(anchor_points, y_anchors, label='锚点样本', color='b', marker='o', s=100, facecolors='r', edgecolors='b', linewidths=2)
plt.plot(x_true, y_pred, label='预测结果', color='green')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc="upper left")
plt.grid(True)
plt.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.15, wspace=0.2, hspace=0.1)
plt.ylim(-1.3, 3)
plt.text(1.5, -1.9, fig_title, ha='center', va='center')
plt.show()
