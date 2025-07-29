import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# 1. 数据准备 ---

def load_dataset(data_path, test_per=0.2):
    # 读取数据
    raw_data = pd.read_csv(data_path)
    raw_data = raw_data.sample(frac=1)
    data = raw_data.iloc[:, 0:4].to_numpy(dtype=np.float32)

    # 标签处理
    label = raw_data['label'].to_numpy()
    encoder = LabelBinarizer()
    label = encoder.fit_transform(label)

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
        self.w = torch.randn(size=[4, 3], requires_grad=True)
        self.b = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

    def __call__(self, x):
        return torch.mm(x, self.w) + self.b


# 损失函数
def loss(y_pred, y_true):
    return F.cross_entropy(y_pred, torch.argmax(y_true, 1)).mean()

# 准确率
def accuracy(y_pred, y_true):
    correct_pred = torch.eq(torch.argmax(y_pred, 1), torch.argmax(y_true, 1))
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
train_data, train_label, test_data, test_label = load_dataset("../datasets/iris.data-tri_class.txt")
dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)


# 4. 创建模型 ---
model = Model()
ls = []
accs = []


# 5. 训练 ---
for _ in range(50):
    for batch_data, batch_label in dataloader:
        l, acc = train_step(model, batch_data, batch_label, 0.5)
        ls.append(l)
        accs.append(acc)
        print(l, acc)


test_acc = accuracy(model(torch.tensor(test_data)), torch.tensor(test_label))
print("test acc:", test_acc.detach().data.numpy())
plt.plot(ls)
plt.plot(accs)
plt.legend(['loss', 'acc'])
plt.show()
