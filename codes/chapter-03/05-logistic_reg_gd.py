from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


np.random.seed(12)


class Model:
    """逻辑回归模型"""

    def __init__(self):
        # 权重和偏置
        self.w = np.array([[1.0], [3.0]])
        self.b = np.array(0.0)

    def __call__(self, x):
        return 1 / (1 + np.exp(-(x @ self.w + self.b)))

    def grad(self, x, y):
        """计算梯度"""
        y_pred = self(x)
        error = y_pred - y
        w_grad = np.dot(x.T, error) / len(x)  # 权值梯度
        b_grad = np.mean(error)  # 偏置梯度
        return w_grad, b_grad


def loss(y_pred, y_true):
    """损失函数"""
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_pred, y_true):
    """准确率"""
    correct_pred = np.equal(np.round(y_pred), y_true)
    return np.mean(correct_pred)


# 训练步
def train_step(model, x, y, learning_rate):
    y_pred = model(x)
    current_loss = loss(y_pred, y)
    w_grad, b_grad = model.grad(x, y)

    # 更新参数
    model.w -= learning_rate * w_grad
    model.b -= learning_rate * b_grad

    acc = accuracy(y_pred, y)
    return current_loss, acc


if __name__ == '__main__':
    # --- 数据准备
    data, label = make_blobs(
        n_samples=150,
        n_features=2,
        centers=2,              # 类别数量
        cluster_std=1.0,        # 每个类别的高斯分布标准差
        center_box=(-5.0, 5.0),  # 每个类别中心的边界范围
        random_state=42         # 随机种子
    )

    # 数据标准化
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std

    # 划分训练集和测试集
    train_data, test_data, train_label, test_label = train_test_split(
        data, label,
        test_size=0.2,
        random_state=42
    )

    train_label = train_label.reshape(-1, 1)  # 确保标签的形状是 (n_samples, 1)
    test_label = test_label.reshape(-1, 1)

    # --- 创建模型
    model = Model()
    ls = []
    accs = []

    # 记录原始损失和准确率
    pred_0 = model(train_data)
    l, acc = loss(pred_0, train_label), accuracy(pred_0, train_label)
    ls.append(l)
    accs.append(acc)

    # --- 训练
    for epoch in range(300):
        l, acc = train_step(model, train_data, train_label, 0.05)
        ls.append(l)
        accs.append(acc)
        print(f'Epoch {epoch+1}, Loss: {l:.4f}, Accuracy: {acc:.4f}')

    # --- 可视化
    plt.figure(figsize=(12, 5))

    # 绘制损失和准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(ls, label='损失', marker='o',  markevery=50)
    plt.plot(accs, label='正确率', marker='^', markevery=50)
    plt.legend(prop={'family': 'SimSun'}, loc='upper left')
    plt.xlabel('Epochs')
    plt.title('损失与准确率的变化情况', fontname="SimSun")
    plt.ylim(0, 1.1)

    plt.subplot(1, 2, 2)

    # 绘制决策边界
    x_min, x_max = test_data[:, 0].min() - 1, test_data[:, 0].max() + 1
    y_min, y_max = test_data[:, 1].min() - 1, test_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    pred = model(grid_data)
    zz = pred.reshape(xx.shape)

    # 根据预测的概率绘制颜色渐变
    # plt.contourf(xx, yy, zz, levels=np.linspace(0, 1, 101), cmap='coolwarm', alpha=0.6)
    plt.pcolormesh(xx, yy, zz, cmap='coolwarm')              # 绘制决策空间

    # 绘制数据点
    # plt.scatter(test_data[test_label.flatten() == 0][:, 0],
    #             test_data[test_label.flatten() == 0][:, 1], label='Class 0', edgecolors='w')
    # plt.scatter(test_data[test_label.flatten() == 1][:, 0],
    #             test_data[test_label.flatten() == 1][:, 1], label='Class 1', edgecolors='w')

    plt.scatter(train_data[train_label.flatten() == 0][:, 0],
                train_data[train_label.flatten() == 0][:, 1], label='Class 0', edgecolors='w')
    plt.scatter(train_data[train_label.flatten() == 1][:, 0],
                train_data[train_label.flatten() == 1][:, 1], label='Class 1', edgecolors='w')


    plt.title('决策边界与训练数据', fontname="SimSun")
    plt.legend()

    plt.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.06, wspace=0.15, hspace=0.1)

    plt.show()
