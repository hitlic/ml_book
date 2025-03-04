import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

np.random.seed(12)

# 生成数据集
data, label = make_blobs(
    n_samples=100,
    n_features=2,
    centers=3,               # 类别数量
    cluster_std=1.0,         # 每个类别的高斯分布标准差
    center_box=(-5.0, 5.0),  # 每个类别中心的边界范围
    random_state=42          # 随机种子
)

# 数据标准化
# mean = data.mean(axis=0)
# std = data.std(axis=0)
# data = (data - mean) / std

class Model:
    """感知机模型"""
    def __init__(self, feat_dim, learning_rate):
        self.w = np.random.random(feat_dim)  # 初始化权重
        self.b = 0.0  # 初始化偏置
        self.lr = learning_rate  # 学习率

    def __call__(self, x):
        return np.dot(x, self.w) + self.b   # 批量计算

    def step(self, x, y):
        if self(x) * y < 0:  # 如果样本被误分类
            self.w += self.lr * y * x       # 更新权重
            self.b += self.lr * y           # 更新偏置
            return True
        return False


# 定义一对多策略中的每个分类器
models = []
for i in range(3):
    model = Model(2, 0.1)  # 创建模型
    models.append(model)


# 训练模型（针对每个类别）
max_iters = 100  # 最大迭代次数
for model_idx in range(3):
    model = models[model_idx]
    current_label = (label == model_idx)  # 训练该类别的模型，当前类别为正类，其他类别为负类
    current_label = np.where(current_label, 1, -1)  # 将标签转换为1和-1
    for i in range(max_iters):
        misclassified = False
        for x, y in zip(data, current_label):
            if model.step(x, y):
                misclassified = True
        if not misclassified:
            print(f"第 {i} 次迭代后模型 {model_idx} 已收敛")
            break


# 可视化决策边界
xx, yy = np.meshgrid(np.linspace(-7, 7, 1000), np.linspace(-7, 7, 1000))
Z = np.zeros((xx.shape[0], xx.shape[1], 3))

# 对每个模型进行预测并绘制决策边界
for model_idx, model in enumerate(models):
    # 使用所有网格点进行预测
    Z[..., model_idx] = np.sign(model(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape))

# 绘制分类结果
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z[..., 0], alpha=0.45, cmap=plt.cm.GnBu)
plt.contourf(xx, yy, Z[..., 1], alpha=0.45, cmap=plt.cm.GnBu)
plt.contourf(xx, yy, Z[..., 2], alpha=0.45, cmap=plt.cm.GnBu)

# 绘制样本点
plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.coolwarm, edgecolors='k', s=50)
plt.xlim(-7, 7)
plt.ylim(-7, 7)
# plt.title("一对多策略", fontname="SimSun")

plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.06, wspace=0.2, hspace=0.1)

plt.show()
