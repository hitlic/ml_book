import numpy as np

def linregress(x, y):
    # 添加偏置项
    X = np.column_stack((x, np.ones_like(x)))
    # 使用最小二乘法计算回归系数
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # 式（13）
    return W

# 示例数据
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 5, 7, 9])

# 计算回归系数
W = linregress(x, y)
print("回归系数:", W)

# 进行预测
y_pred = np.dot(np.column_stack((x, np.ones_like(x))), W)
print("预测结果:", y_pred)
