import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist


# 设置图片中文字体，确保能正确显示中文和数学符号
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 22
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'



def classification_dataset():
    """构造数据集"""
    class1_1 = multivariate_normal([1, 1], 0.5 * np.eye(2), 25)
    class1_2 = multivariate_normal([5, 5], 0.5 * np.eye(2), 25)
    class2_1 = multivariate_normal([1, 5], 0.5 * np.eye(2), 25)
    class2_2 = multivariate_normal([5, 1], 0.5 * np.eye(2), 25)
    X = np.concatenate([class1_1, class1_2, class2_1, class2_2], axis=0)
    y = np.array([0] * 50 + [1] * 50)
    return X, y


def rbf_feats(feats, anchors_std, sigma):
    """使用rbf核构造核特征向量"""
    # 数据标准化
    feats_std = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    # 计算距离距阵
    dm = cdist(feats_std, anchors_std, "euclidean")
    return np.exp(- dm / (2 * sigma ** 2))  # rbf特征


if __name__ == "__main__":
    np.random.seed(42)

    X, y = classification_dataset()

    sigma = 1.0

    # 选择锚点并标准化
    anchors = np.array([[1,1], [1,5], [5,1], [5,5]])
    anchors_std = (anchors - anchors.mean(axis=0)) / anchors.std(axis=0)

    X_rbf = rbf_feats(X, anchors_std, sigma)
    model = LogisticRegression()
    model.fit(X_rbf, y)

    ypred = model.predict(X_rbf)
    print(f'正确率{np.mean(ypred == y)}')


    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    plt.figure(figsize=(10, 8))
    # 决策空间
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    X_plot_raw = np.c_[xx.ravel(), yy.ravel()]
    X_plot = rbf_feats(X_plot_raw, anchors_std, sigma)
    Z = model.predict_proba(X_plot)[:,0,np.newaxis].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='magma', alpha=1.0)              # 绘制决策空间

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')    # 绘制样本点
    plt.scatter(anchors[:, 0], anchors[:, 1], marker='<', c='r', s=100, label='锚点')  # 绘制锚点
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc='upper left')
    plt.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.1, wspace=0.2, hspace=0.1)
    plt.show()
