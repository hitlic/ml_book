from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 数据
X, y = load_breast_cancer(return_X_y=True)

# 模型
clf = LogisticRegression(solver="liblinear", random_state=0)

# 学习（拟合）
clf.fit(X, y)

# 预测
pred = clf.predict_proba(X)[:, 1]

# 评价
print('正确率:\t', accuracy_score(y, pred.round()))
