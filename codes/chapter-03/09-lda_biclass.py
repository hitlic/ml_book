import numpy as np
from sklearn.datasets import make_blobs


# 线性判别分析二分类算法
def lda_binary_classification(X, y):
    # 计算每个类别的均值向量
    mean_vectors = []
    for cl in range(2):
        mean_vectors.append(np.mean(X[y == cl], axis=0))

    # 计算全局均值向量
    global_mean = np.mean(X, axis=0)

    # 计算类内散度矩阵 S_W
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for cl, mv in zip(range(2), mean_vectors):
        class_sc_mat = np.zeros((X.shape[1], X.shape[1]))  # 每个类别的散度矩阵
        for row in X[y == cl]:
            row, mv = row.reshape(X.shape[1], 1), mv.reshape(X.shape[1], 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat

    # 计算类间散度矩阵 S_B
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for cl, mean_vec in enumerate(mean_vectors):
        n = X[y == cl].shape[0]
        mean_vec = mean_vec.reshape(X.shape[1], 1)
        global_mean = global_mean.reshape(X.shape[1], 1)
        S_B += n * (mean_vec - global_mean).dot((mean_vec - global_mean).T)

    # 对 S_W 进行平方根分解
    L = np.linalg.cholesky(S_W)
    L_inv = np.linalg.inv(L)

    # 求解广义特征值问题
    S_B_transformed = L_inv.dot(S_B).dot(L_inv.T)
    eig_vals, eig_vecs = np.linalg.eig(S_B_transformed)

    # 取最大特征值对应的特征向量
    top_eig_vec = eig_vecs[:, np.argmax(eig_vals)]

    # 恢复投影向量
    w = L_inv.T.dot(top_eig_vec)

    # 归一化投影向量
    w = w / np.linalg.norm(w)

    return w


if __name__ == '__main__':
    # 生成数据集
    np.random.seed(12)
    data, label = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,              # 类别数量
        cluster_std=1.0,        # 每个类别的高斯分布标准差
        center_box=(-5.0, 5.0),  # 每个类别中心的边界范围
        random_state=42         # 随机种子
    )

    # 调用LDA算法求解投影方向
    w = lda_binary_classification(data, label)
    print(w)
