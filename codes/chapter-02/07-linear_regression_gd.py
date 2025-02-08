from matplotlib import pyplot as plt
import numpy as np


def linear_data_gen(w=3.0, b=2.0, num=1000):
    """随机生成线性回归数据集 y = w * x + b"""
    x = np.random.randn(num, 1)
    noise = np.random.randn(num, 1)
    y = x * w + b + noise
    return np.column_stack([x, np.ones_like(x)]), y


# 模型定义
class Model:
    def __init__(self):
        self.w = np.array([[5.0], [0.0]])  # 初始化，实际使用随机初始化

    def __call__(self, x):
        return x @ self.w


def grad(w, x, y):
    """计算梯度"""
    xt = np.transpose(x)
    return 2 * xt @ x @ w - 2*xt @ y


def step(model, x, y, learning_rate):
    """梯度下降参数更新"""
    gd = grad(model.w, x, y)/len(x)# 利用均方误差作为损失函数，式（12）
    new_w = model.w - learning_rate * gd
    model.w = new_w


# 数据准备
target_w = 3.0
target_b = 2.0
x, y = linear_data_gen(target_w, target_b)

# 创建模型
model = Model()

y_old = model(x)  # 原始模型的预测


# 训练
ws, bs = [], []  # 训练过程中所有 w 和 s
epochs = range(15)

for epoch in epochs:
    ws.append(model.w[0][0])
    bs.append(model.w[1][0])

    step(model, x, y, 0.1)
    print(f'Epoch {epoch:<2}: W={ws[-1]:<3.2} b={bs[-1]:<3.2}')


# 画图
plt.figure(figsize=(12, 4))
font = {'family': 'simsun'}  # 中文字体

# 左图
fig1 = plt.subplot(131)
fig1.scatter(x[:,[0]], y, c='b', marker='o', s=4)
fig1.scatter(x[:,[0]], y_old, c='r', marker='o', s=4)
fig1.set_title("训练前", fontdict=font)

# 中图
fig2 = plt.subplot(132)
fig2.plot(epochs, ws, 'y')
fig2.plot(epochs, bs, 'm')
fig2.plot([target_w] * len(epochs), 'y--', [target_b] * len(epochs), 'm--')
fig2.legend(['w', 'b', 'target w', 'target b'])
fig2.set_title("训练中", fontdict=font)

# 右图
fig3 = plt.subplot(133)
fig3.scatter(x[:,[0]], y, c='b', marker='o', s=4)
fig3.scatter(x[:,[0]], model(x), c='g', marker='o', s=4)
fig3.set_title("训练后", fontdict=font)

plt.show()
