import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# 设置图片字体
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Times New Roman'
# plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
# plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16


np.random.seed(12)

# 生成二维高斯分布的二分类数据集
data, label = make_blobs(
    n_samples=100,          # 样本数量
    n_features=2,           # 特征数量（二维）
    centers=2,              # 类别数量（二分类）
    cluster_std=1.0,        # 每个类别的高斯分布标准差
    center_box=(-5.0, 5.0),  # 每个类别中心的边界范围
    random_state=42         # 随机种子
)

# 标签处理：转换为-1和1
label[label == 0] = -1

# 数据标准化
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std

# 可视化函数


def plot_decision_boundary(model, iteration, max_iters, color_map):
    # 计算决策边界的直线
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3

    # 解方程 w1*x1 + w2*x2 + b = 0，计算x2 = (-w1*x1 - b) / w2
    x1_vals = np.linspace(x_min, x_max, 100)
    x2_vals = (-model.w[0] * x1_vals - model.b) / model.w[1]

    # 计算当前迭代的颜色
    color_ratio = iteration / max_iters
    color = color_map(color_ratio)  # 根据迭代次数调整颜色

    # 绘制直线
    plt.plot(x1_vals, x2_vals, color=color, label=f'Iteration {iteration + 1}' if iteration % 10 == 0 else "")

# 感知机模型


class Model:
    def __init__(self, feat_dim, learning_rate):
        self.w = np.random.random(feat_dim)  # 初始化权重
        self.b = 0.0  # 初始化偏置
        self.lr = learning_rate  # 学习率

    def __call__(self, x):
        return np.dot(self.w, x) + self.b   # 模型计算

    def step(self, x, y):
        if self(x) * y < 0:  # 如果样本被误分类
            self.w += self.lr * y * x       # 更新权重
            self.b += self.lr * y           # 更新偏置
            return True  # 标记有更新
        return False  # 没有更新


# 训练模型
model = Model(2, 0.1)   # 初始化模型
max_iters = 100         # 最大迭代次数

# 创建一个颜色映射
color_map = plt.cm.get_cmap('YlGnBu')

# 设置绘图
plt.figure(figsize=(8, 6))


# 初始绘制决策边界
plot_decision_boundary(model, 0, max_iters, color_map)

update_times = 0
for i in range(max_iters):
    misclassified = False  # 标记是否有误分类样本
    for x, y in zip(data, label):
        if model.step(x, y):  # 如果有误分类，更新权重并绘制决策边界
            misclassified = True
            plot_decision_boundary(model, update_times + 1, 8, color_map)  # 训练后画出当前的决策边界
            update_times += 1
    if not misclassified:  # 如果没有误分类样本，提前终止
        print(f"Early stopping at iteration {i}")
        break

# 输出模型参数
print("Model weights (w):", model.w)
print("Model bias (b):", model.b)

print(update_times)

# 绘制数据点
plt.scatter(data[label == -1, 0], data[label == -1, 1], color='red', label='Class -1', marker='s')
plt.scatter(data[label == 1, 0], data[label == 1, 1], color='blue', label='Class 1')

# 设置坐标轴范围
plt.xlim(-3, 3)
plt.ylim(-3, 3)
# plt.grid(True)

plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.06, wspace=0.2, hspace=0.1)

plt.show()
