"""
C4.5决策树实现（不含缺失值处理）
"""
from collections import Counter
import numpy as np
from utils import info_gain, info_gain_continuous, is_samples_same, draw_tree, is_discrete


class Node:
    def __init__(self, labels, feat):
        self.labels = labels     # 当前节点的标签
        self.feat = feat         # 当前节点检测的特征
        self.is_leaf = False     # 是否叶节点
        self.class_label = None  # 类别
        self.discret= None

    def get_children(self):
        """用于绘制决策树"""
        if self.discret is False:
            return [(f'<={self.cut_point_value:.4}', self.l_child), (f'>{self.cut_point_value:.4}', self.r_child)]
        else:
            return self.children.items()

    @property
    def id(self):
        """用于绘制决策树"""
        return str(id(self))

    def to_str(self):
        """用于绘制决策树"""
        return f'{"Class: "+self.class_label if self.is_leaf else self.feat}\n{len(self.labels)} | {dict(Counter(self.labels))}'


class DiscreteNode(Node):
    def __init__(self, datas, feat):
        """离散特征节点"""
        super().__init__(datas, feat)
        self.children = {}      # 子节点字典；key为特征取值，value为Node对象
        self.discret = True

    def add_child(self, feat_value, child_node):
        """增加子节点"""
        self.children[feat_value] = child_node

    def condition_check(self, value):
        """根据特征值检查子节点"""
        return self.children[value]


class ContinuousNode(Node):
    def __init__(self, datas, feat, cut_point_value):
        """连续特征节点"""
        super().__init__(datas, feat)
        self.l_child = None
        self.r_child = None
        self.cut_point_value = cut_point_value
        self.discret = False

    def add_l_child(self, child_node):
        self.l_child = child_node

    def add_r_child(self, child_node):
        self.r_child = child_node

    def condition_check(self, feat_value):
        if feat_value <= self.cut_point_value:
            return self.l_child
        else:
            return self.r_child


def best_feat(feat_set, datas, labels, criterion):
    """寻找最优特征及相应的信息增益或信息增益率"""

    max_criterion_value = -1        # 最大信息增益或信息增益率
    best_f = None                   # 最优特征
    best_cut_point = None           # 最优切分点，离散特征下取值为None
    for feat in feat_set:
        feat_values = datas[feat]
        # 计算信息增益或信息增益率
        if is_discrete(datas[feat]):  # 离散特征
            criterion_value = info_gain(labels, feat_values, criterion != 'IG')
            cut_point = None
        else:                         # 连续特征
            criterion_value, cut_point = info_gain_continuous(labels, feat_values, criterion != 'IG')

        if criterion_value > max_criterion_value:
            max_criterion_value = criterion_value
            best_f = feat
            best_cut_point = cut_point
    return best_f, best_cut_point


def build_tree(feat_set, datas, labels, criterion):
    """构建决策树"""
    def leaf_node(labels):
        """创建叶节点"""
        node = Node(labels, None)
        node.class_label = Counter(labels).most_common(1)[0][0]
        node.is_leaf = True
        return node

    if len(np.unique(labels)) == 1 or len(feat_set) == 0 or is_samples_same(datas):
        # 当标签labels中只包含一种类别、特征集为空，或者所有样本特征相同时，当前节点为叶节点
        return leaf_node(labels)
    else:
        feat, cut_point = best_feat(feat_set, datas, labels, criterion)
        # 当根据最优特征划分后信息增益或信息增益率不增加时，创建叶节点
        # if criterion_value <= best_criterion_value:
        #     return leaf_node(labels)

        if cut_point is None:           # ------ 离散特征划分子节点
            feat_set.remove(feat)  # 离散特征只划分一次
            node = DiscreteNode(labels, feat)
            # 基于feat构建子树
            for feat_value in np.unique(datas[feat]):
                mask = datas[feat]==feat_value
                datas_ = datas[mask]    # 子树数据
                labels_ = labels[mask]  # 子树标签
                node.add_child(feat_value, build_tree(feat_set, datas_, labels_, criterion))
            return node
        else:                           # ------ 连续特征划分子节点
            node = ContinuousNode(labels, feat, cut_point)
            # 左分支，特征取值小于等于cut point
            mask_l = datas[feat] <= cut_point
            if mask_l.sum() == 0:
                pass
            datas_l, labels_l = datas[mask_l], labels[mask_l]
            node.add_l_child(build_tree(feat_set, datas_l, labels_l, criterion))
            # 右分支，特征取值大于cut point
            mask_r = datas[feat] > cut_point
            if mask_r.sum() == 0:
                pass
            datas_r, labels_r = datas[mask_r], labels[mask_r]
            node.add_r_child(build_tree(feat_set, datas_r, labels_r, criterion))
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


def prune_tree(node):
    """
    对决策树进行自底向上的C4.5剪枝（误差估计法）。
    node: 决策树节点（Node/DiscreteNode/ContinuousNode），需有labels属性。
    剪枝原则参考C4.5算法伪代码。
    """

    # 递归终止条件：叶节点
    if node.is_leaf:
        # 叶节点的错误数
        m = sum(np.array(node.labels) != node.class_label)
        return m, 1  # 错误数, 叶节点数

    # 递归计算所有子节点的误差
    if node.discret:
        # 离散节点
        subtree_error = 0
        leaf_count = 0
        for child in node.children.values():
            m_i, l_i = prune_tree(child)
            subtree_error += m_i
            leaf_count += l_i
    else:
        # 连续节点
        m_left, l_left = prune_tree(node.l_child)
        m_right, l_right = prune_tree(node.r_child)
        subtree_error = m_left + m_right
        leaf_count = l_left + l_right

    # 该节点自身剪枝后的误差（变为叶节点）
    # 预测类别为最多类别
    most_common_label = Counter(node.labels).most_common(1)[0][0]
    m = sum(np.array(node.labels) != most_common_label)
    E_subtree = subtree_error + 0.5 * leaf_count
    E_pruned = m + 0.5

    # 剪枝条件
    if E_pruned <= E_subtree:
        # 剪枝为叶节点
        node.is_leaf = True
        node.class_label = most_common_label
        # 移除子节点
        if node.discret:
            node.children = {}
        else:
            node.l_child = None
            node.r_child = None
        return m, 1  # 剪枝后该节点是叶节点
    else:
        # 不剪枝，返回子树误差统计
        return subtree_error, leaf_count



if __name__ == '__main__':
    import pandas as pd

    # data = {
    #     '天气':['晴', '晴', '阴', '雨', '雨', '雨', '阴', '晴', '晴', '雨', '晴', '阴', '阴', '雨', '晴'],
    #     '温度':['热', '热', '热', '中', '冷', '冷', '冷', '中', '冷', '中', '中', '中', '热', '中', '中'],
    #     '运动': ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是', '是', '是', '是', '否', '是']
    # }
    # data = pd.DataFrame(data)
    data = pd.read_csv('../datasets/iris.csv')

    datas, labels = data.iloc[:, :-1], data.iloc[:, -1]

    dt = DecisionTree(criterion='IGR')  # IG: 信息增益；IGR: 信息增益率
    dt.fit(datas, labels)
    prune_tree(dt.tree)  # 对决策树进行剪枝
    print('预测 \t 标签')
    for (row_id, data), l in zip(datas.iterrows(), labels):
        print(dt.predict(data), '\t', l)

    draw_tree(dt.tree)
