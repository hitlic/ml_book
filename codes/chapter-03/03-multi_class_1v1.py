import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16


np.random.seed(1234)

# 生成三分类的二维数据集
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


# 定义模型类
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


# 定义一对一策略中的分类器
models = []
class_pairs = [(0, 1), (0, 2), (1, 2)]  # 三个类别对

for (c1, c2) in class_pairs:
    model = Model(2, 0.1)  # 创建模型
    models.append((model, c1, c2))  # 存储模型和类别对


# 训练模型
max_iters = 100  # 最大迭代次数
for model_idx, (model, c1, c2) in enumerate(models):
    # 根据当前类别对筛选数据
    mask = (label == c1) | (label == c2)
    current_data = data[mask]
    current_label = np.where(label[mask] == c1, 1, -1)  # c1为正类，c2为负类


    for i in range(max_iters):
        misclassified = False
        for x, y in zip(current_data, current_label):
            if model.step(x, y):
                misclassified = True
        if not misclassified:
            print(f"第 {i} 次迭代后模型 {c1}-{c2} 已收敛")
            break

# 可视化决策边界
xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))


# 投票进行分类
def predict_one_vs_one(x):
    votes = np.zeros(3)
    for model, c1, c2 in models:
        prediction = np.sign(model(x))  # 获取当前模型的预测
        if prediction == 1:
            votes[c1] += 1
        else:
            votes[c2] += 1
    return np.argmax(votes)  # 返回投票结果最多的类别


# 对每个网格点进行预测
Z = np.zeros(xx.shape)  # 创建一个二维数组来存储最终类别

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        Z[i, j] = predict_one_vs_one(np.array([xx[i, j], yy[i, j]]))

# 绘制分类结果
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)

# 绘制样本点
plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.coolwarm, edgecolors='k', s=50)
plt.xlim(-7, 7)
plt.ylim(-7, 7)
# plt.title("一对一策略", fontname="SimSun")

plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.06, wspace=0.2, hspace=0.1)

plt.show()
