import numpy as np
from sklearn.datasets import make_blobs

np.random.seed(12)

# 生成数据集
data, label = make_blobs(
    n_samples=100,
    n_features=2,
    centers=2,              # 类别数量
    cluster_std=1.0,        # 每个类别的高斯分布标准差
    center_box=(-5.0, 5.0),  # 每个类别中心的边界范围
    random_state=42         # 随机种子
)

# 标签处理：转换为-1和1
label[label == 0] = -1


# 数据预处理
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean)/std


class Model:
    def __init__(self, feat_dim, learning_rate):
        self.w = np.random.random(feat_dim)
        self.b = np.array([0.0])
        self.lr = learning_rate

    def __call__(self, x):
        return (self.w * x).sum() + self.b

    def step(self, x, y):
        if self(x) * y < 0:
            self.w += self.lr * y * x
            self.b += self.lr * y


# 学习
model = Model(2, 0.1)
for i in range(2):
    for x, y in zip(data, label):
        model.step(x, y)


print(model.w, model.b)

# 性能测试
for x, y in zip(data, label):
    print(model(x), y)
