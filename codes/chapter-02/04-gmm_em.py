"""
高斯混合模型 EM 算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# plt.rcParams['text.usetex'] = True  # 启用LaTeX渲染
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14


def generate_data():
    """生成示例数据"""
    np.random.seed(42)

    # 三个高斯分布的样本
    mean1, cov1 = [2, 2], [[0.1, 0], [0, 0.1]]
    mean2, cov2 = [4, 4], [[0.2, 0], [0, 0.2]]
    mean3, cov3 = [6, 2], [[0.3, 0], [0, 0.3]]

    data1 = np.random.multivariate_normal(mean1, cov1, 100)
    data2 = np.random.multivariate_normal(mean2, cov2, 100)
    data3 = np.random.multivariate_normal(mean3, cov3, 100)

    return np.vstack((data1, data2, data3))


def initialize_parameters(data, K):
    """初始化模型参数"""
    N, D = data.shape
    pi = np.ones(K) / K                               # 均匀初始化混合系数
    mu = data[np.random.choice(N, K, replace=False)]  # 随机选择 K 个点作为均值初始值
    sigma = np.array([np.eye(D) for _ in range(K)])   # 初始化协方差矩阵为单位矩阵
    return pi, mu, sigma


def gaussian_pdf(x, mu, sigma):
    """高斯分布的概率密度函数"""
    D = x.shape[0]
    diff = x - mu
    exp_term = np.exp(-0.5 * diff.T @ np.linalg.inv(sigma) @ diff)
    denominator = np.sqrt((2 * np.pi) ** D * np.linalg.det(sigma))
    return exp_term / denominator


def e_step(data, pi, mu, sigma, K):
    """E步：计算每个样本属于第 k 个高斯分量的概率"""
    N = data.shape[0]
    gamma = np.zeros((N, K))

    for i in range(N):
        probs = np.array([pi[k] * gaussian_pdf(data[i], mu[k], sigma[k]) for k in range(K)])
        gamma[i, :] = probs / np.sum(probs)

    return gamma


def m_step(data, gamma, K):
    """M步：更新模型参数"""
    N = data.shape[0]
    N_k = np.sum(gamma, axis=0)

    # 更新混合系数
    pi = N_k / N

    # 更新均值
    mu = np.array([np.sum(gamma[:, k].reshape(-1, 1) * data, axis=0) / N_k[k] for k in range(K)])

    # 更新协方差矩阵
    sigma = []
    for k in range(K):
        diff = data - mu[k]
        weighted_sum = sum(gamma[i, k] * np.outer(diff[i], diff[i]) for i in range(N))
        sigma.append(weighted_sum / N_k[k])
    sigma = np.array(sigma)

    return pi, mu, sigma


def gmm_em(data, K=3, max_iter=100, tol=1e-4):
    """高斯混合模型 EM 算法"""
    # 初始化参数
    pi, mu, sigma = initialize_parameters(data, K)

    for iteration in range(max_iter):
        # E步
        gamma = e_step(data, pi, mu, sigma, K)
        # M步
        pi_new, mu_new, sigma_new = m_step(data, gamma, K)

        # 检查收敛条件
        if np.linalg.norm(mu_new - mu) < tol:
            print(f"收敛于第 {iteration + 1} 次迭代")
            break

        pi, mu, sigma = pi_new, mu_new, sigma_new

    return pi, mu, sigma, gamma


def plot_gaussian_contours(mus, sigmas, pis):
    """绘制高斯混合分布概率密度"""
    ax = plt.gca()
    x = np.linspace(0, 8, 100)
    y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # 计算混合高斯分布的概率密度
    Z = np.zeros(X.shape)
    for pi, mu, sigma in zip(pis, mus, sigmas):
        rv = multivariate_normal(mean=mu, cov=sigma)
        Z += pi * rv.pdf(pos)
    ax.contourf(X, Y, Z, cmap='Blues', alpha=1.0, levels=20)


# 绘制结果
def plot_results(data, mu, gamma):
    labels = np.argmax(gamma, axis=1)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.scatter(mu[:, 0], mu[:, 1], c='red', marker='^', s=100, label='聚类中心')

    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.legend()

    # plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.06, wspace=0.2, hspace=0.2)
    plt.show()


if __name__ == "__main__":
    data = generate_data()
    K = 3  # 高斯分量数量

    # 运行 EM 算法
    pi, mu, sigma, gamma = gmm_em(data, K)

    # 输出结果
    print("混合系数:", pi)
    print("均值:", mu)
    print("协方差矩阵:", sigma)

    # 绘制聚类结果
    plot_gaussian_contours(mu, sigma, pi)
    plot_results(data, mu, gamma)
