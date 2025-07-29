"""
基本决策树构建算法，只处理离散特征
"""
from collections import Counter
import numpy as np
from utils import info_gain, is_samples_same, draw_tree


class Node:
    def __init__(self, labels, feat):
        self.labels = labels     # 当前节点中所有样本的标签
        self.feat = feat         # 当前节点检测的特征
        self.is_leaf = False     # 是否叶节点
        self.class_label = None  # 叶节点表示的类别
        self.children = {}       # 子节点字典；key为特征取值，value为Node对象
        self.discret = True

    def add_child(self, feat_value, child_node):
        """增加子节点"""
        self.children[feat_value] = child_node

    def condition_check(self, value):
        """根据特征值检查子节点"""
        return self.children[value]

    def get_children(self):
        """用于绘制决策树"""
        return self.children.items()

    @property
    def id(self):
        """用于绘制决策树"""
        return str(id(self))

    def to_str(self):
        """用于绘制决策树"""
        return f'{"Class: "+self.class_label if self.is_leaf else self.feat}\n{len(self.labels)} | {dict(Counter(self.labels))}'


def best_feat(feat_set, datas, labels, criterion):
    """寻找最优特征及相应的信息增益或信息增益率"""
    max_criterion_value = -1
    best_f = None
    for feat in feat_set:
        feat_values = datas[feat]
        criterion_value = info_gain(labels, feat_values, criterion != 'IG')  # 计算信息增益或信息增益率
        if criterion_value > max_criterion_value:
            max_criterion_value = criterion_value
            best_f = feat
    return best_f


def build_tree(feat_set, datas, labels, criterion):
    # 当标签labels中只包含一种类别、特征集为空，或者所有样本特征相同时，当前节点为叶节点
    if len(np.unique(labels)) == 1 or len(feat_set) == 0 or is_samples_same(datas):
        node = Node(labels, None)
        node.class_label = Counter(labels).most_common(1)[0][0]
        node.is_leaf = True
        return node
    else:
        feat = best_feat(feat_set, datas, labels, criterion)
        node = Node(labels, feat)
        # 基于feat构建子树
        feat_set.remove(feat)    # 子树特征集
        for feat_value in np.unique(datas[feat]):
            mask = datas[feat]==feat_value
            datas_ = datas[mask]    # 子树数据
            labels_ = labels[mask]  # 子树标签
            node.add_child(feat_value, build_tree(feat_set, datas_, labels_, criterion))
        return node


class DecisionTree:
    def __init__(self, criterion):
        assert criterion in ['IG', 'IGR']
        self.criterion = criterion
        self.tree = None        # 决策树
        self.criterion = criterion

    def fit(self, datas, labels):
        feat_set = list(datas.columns)
        self.tree = build_tree(feat_set, datas, labels, self.criterion)

    def predict(self, data):
        node = self.tree
        while True:
            if node.is_leaf:
                return node.class_label
            node = node.condition_check(data[node.feat])


if __name__ == '__main__':
    import pandas as pd

    data = {
        '天气':['晴', '晴', '阴', '雨', '雨', '雨', '阴', '晴', '晴', '雨', '晴', '阴', '阴', '雨', '晴'],
        '温度':['热', '热', '热', '中', '冷', '冷', '冷', '中', '冷', '中', '中', '中', '热', '中', '中'],
        '运动': ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是', '是', '是', '是', '否', '是']
    }
    data = pd.DataFrame(data)

    datas, labels = data.iloc[:, :-1], data.iloc[:, -1]

    dt = DecisionTree(criterion='IG')  # IG: 信息增益；IGR: 信息增益率
    dt.fit(datas, labels)
    print('预测 \t 标签')
    for (row_id, data), l in zip(datas.iterrows(), labels):
        print(dt.predict(data), '\t', l)

    draw_tree(dt.tree)
