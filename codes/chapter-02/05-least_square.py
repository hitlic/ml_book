import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置字体
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Times New Roman'
# plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
# plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 13


# 生成样本数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100个样本，1个特征
y = 4 + 3 * X + np.random.randn(100, 1)  # 目标变量，带有噪声



# === 使用最小二乘法解线性回归模型
X_b = np.c_[np.ones((100, 1)), X]  # 加入偏置项
w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # 计算最优参数
print("最优参数：\n", w)



# === 使用scikit-learn的线性回归模型进行验证
model = LinearRegression()
model.fit(X, y)
print("sklearn线性回归系数：", model.intercept_, model.coef_)



# 绘制数据与模型
plt.scatter(X, y, color='blue', label='样本数据点')
plt.plot(X, model.predict(X), color='red', label='拟合模型')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend(prop={'family': 'SimSun'})

# plt.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.1, wspace=0.2, hspace=0.2)

plt.show()
