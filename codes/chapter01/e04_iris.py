from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 数据
X, y = load_iris(return_X_y=True)

# 模型
clf = LogisticRegression(solver="liblinear")

# 学习（拟合）
clf.fit(X, y)

# 预测
pred = clf.predict_proba(X)

print('正确率:\t', accuracy_score(y, pred.argmax(axis=1)))