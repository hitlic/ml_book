import torch
from matplotlib import pyplot as plt

# 1. 数据生成函数 ---
def linear_data_gen(w=3.0, b=2.0, num=1000):
    """
    随机生成线性回归数据集 y = w * x + b
    """
    x = torch.randn(size=[num])
    noise = torch.randn(size=[num])
    y = x * w + b + noise
    return x, y


# 2. 模型定义 ---
class Model:
    def __init__(self):
        self.w = torch.tensor(5.0, requires_grad=True)  # 初始化，实际使用随机初始化
        self.b = torch.tensor(0.0, requires_grad=True)

    def __call__(self, x):
        return self.w * x + self.b


def loss(y_pred, y_true):
    return torch.mean(torch.pow(y_pred - y_true, 2))


def train_step(model, x, y, learning_rate):
    # 前向计算
    current_loss = loss(model(x), y)
    # 反向计算
    current_loss.backward()
    with torch.no_grad():
        # 梯度下降
        model.w.data -= learning_rate * model.w.grad
        model.b.data -= learning_rate * model.b.grad
        # 梯度置0
        model.w.grad.zero_()
        model.b.grad.zero_()
    return current_loss

# 3. 数据准备 ---
target_w = 3.0
target_b = 2.0
x, y = linear_data_gen(target_w, target_b)

# 4. 创建模型 ---
model = Model()
y_old = model(x)  # 原始模型的预测


# 5. 训练 ---
ws, bs = [], []  # 训练过程中所有 w 和 s
epochs = range(15)

for epoch in epochs:
    ws.append(model.w.item())
    bs.append(model.b.item())

    l = train_step(model, x, y, 0.1)
    print(f'Epoch {epoch:<2}: W={ws[-1]:<3.2} b={bs[-1]:<3.2}, loss={l}')


# 6. 画图 ---
plt.figure(figsize=(12, 4))
font = {'family': 'simsun'}  # 中文字体

# 左图
fig1 = plt.subplot(131)
fig1.scatter(x, y, c='b', marker='o', s=4)
fig1.scatter(x, y_old.data, c='r', marker='o', s=4)
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
fig3.scatter(x, y, c='b', marker='o', s=4)
fig3.scatter(x, model(x).data, c='g', marker='o', s=4)
fig3.set_title("训练后", fontdict=font)

# 显示图
plt.show()
