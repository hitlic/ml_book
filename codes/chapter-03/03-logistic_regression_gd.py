import torch
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from torch.nn import functional as F


# 1. 数据准备 ---
def load_dataset(data_path, test_per=0.2):
    # 读取数据
    raw_data = pd.read_csv(data_path)
    raw_data = raw_data.sample(frac=1)
    data = raw_data.iloc[:, 0:4].to_numpy(dtype=np.float32)

    # 标签处理
    label = raw_data['label'].to_numpy()
    label_set = np.unique(label).tolist()
    label = np.array([[label_set.index(l)] for l in label])

    # 数据预处理
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean)/std

    test_start = round(len(data) * (1 - test_per))

    # 数据划分
    test_data = data[test_start:]
    test_label = label[test_start:]
    data = data[0:test_start]
    label = label[0:test_start]
    return data, label, test_data, test_label


# 2. 模型定义 ---
class Model:
    def __init__(self):
        self.w = torch.tensor([[0.3], [0.04], [-0.2], [0.06]], requires_grad=True)  # 初始化，实际使用随机初始化
        self.b = torch.tensor(0.0, requires_grad=True)

    def __call__(self, x):
        return 1/(1 + torch.exp(-(torch.mm(x, self.w) + self.b)))

# 损失函数
def loss(y_pred, y_true):
    # return torch.mean(torch.pow(y_true - y_pred, 2))
    return F.binary_cross_entropy(y_pred, y_true)


# 准确率
def accuracy(y_pred, y_true):
    correct_pred = torch.eq(torch.round(y_pred), y_true)
    return torch.mean(correct_pred.float())


# 训练步
def train_step(model, x, y, learning_rate):
    current_loss = loss(model(x), y)
    current_loss.backward()
    with torch.no_grad():
        model.w.data -= learning_rate * model.w.grad
        model.b.data -= learning_rate * model.b.grad

        model.w.grad.zero_()
        model.b.grad.zero_()

        acc = accuracy(model(x), y)

    return current_loss.item(), acc.item()


# 3. 加载数据 ---
train_data, train_label, test_data, test_label = load_dataset("../datasets/iris.data-bi_class.txt")

train_data = torch.tensor(train_data, dtype=torch.float)
train_label = torch.tensor(train_label, dtype=torch.float)
test_data = torch.tensor(test_data, dtype=torch.float)
test_label = torch.tensor(test_label, dtype=torch.float)

# 4. 创建模型 ---
model = Model()
ls = []
accs = []
# 5. 训练 ---
for epoch in range(100):
    l, acc = train_step(model, train_data, train_label, 0.05)
    ls.append(l)
    accs.append(acc)
    print(l, '\t',acc)

plt.plot(ls)
plt.plot(accs)
plt.legend(['loss', 'acc'])
plt.show()
