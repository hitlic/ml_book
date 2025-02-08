import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# 加载 diabetes 数据集
X, y = datasets.load_diabetes(return_X_y=True)

# 为了画图，仅使用一个特征
X = X[:, np.newaxis, 2]

# 训练集/测试集划分
X_train = X[:-20]
X_test = X[-20:]
y_train = y[:-20]
y_test = y[-20:]

regr = linear_model.LinearRegression()           # 线性回归模型
regr.fit(X_train, y_train)     # 模型拟合
y_pred = regr.predict(X_test)  # 预测

print(f"系数: {regr.coef_},  截距：{regr.intercept_}")
mse = mean_squared_error(y_test, y_pred)
print(f"平均绝对误: {mse}")

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color="blue")
plt.show()
