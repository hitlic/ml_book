import numpy as np
from collections import Counter
import graphviz
import pandas as pd
from pandas.api.types import is_string_dtype, is_bool


def prob_dist(values):
    """统计一组离散取值的频率作为概率"""
    counter = Counter(values)
    return np.array(list(counter.values()), dtype=np.float32)/counter.total()


def entropy(values):
    """概率分布的熵"""
    probs = prob_dist(values)
    return - (probs * np.log2(probs)).sum()


def conditional_entropy(values, cond_values):
    """条件熵"""
    size = len(values)
    ce = 0
    for cond in np.unique(cond_values):
        cond_mask = cond_values==cond
        p = cond_mask.sum()/size
        ce += p * entropy(values[cond_mask])
    return ce


def info_gain(values, cond_values, ratio=False):
    """信息增益（信息增益率）
    Args:
        values: 样本标签
        cond_values: 样本特征值
        ratio: 是否计算信息增益率
    """
    # 信息增
    ig = entropy(values) - conditional_entropy(values, cond_values)
    if not ratio:
        return ig
    # 计算信息增益率
    iv = entropy(cond_values)
    return ig / iv if iv != 0 else 0


def info_gain_continuous(values, cond_values, ratio=False):
    """连续取值属性的信息增益率（或信息增益率）最大的切分点
    Args:
        values: 样本标签
        cond_values: 样本特征值
        ratio: 是否计算信息增益率
    """
    size = len(values)
    cut_values = np.array(sorted(list(cond_values)))
    cut_points = (cut_values[:-1] + cut_values[1:])/2
    ent = entropy(values)
    max_criterion_value = -1
    best_cp = None
    for cp in cut_points:
        mask_l = cond_values <= cp
        mask_r = np.logical_not(mask_l)
        cond_ent = mask_l.sum()/size * entropy(values[mask_l]) + mask_r.sum()/size * entropy(values[mask_r])  # 条件熵
        criterion_value = ent - cond_ent  # 信息增益
        if ratio:
            if mask_l.sum() == 0 or mask_r.sum() == 0:
                continue
            iv = - mask_l.sum()/size * np.log2(mask_l.sum()/size) - mask_r.sum()/size * np.log2(mask_r.sum()/size)  # 固有值
            criterion_value = criterion_value / iv if iv != 0 else 0  # 信息增益率
        if max_criterion_value < criterion_value:
            max_criterion_value = criterion_value
            best_cp = cp
    return max_criterion_value, best_cp


def feat_gini(values):
    """特征gini指数"""
    unique_values = values.unique()
    value = 0
    size = len(values)
    for v in unique_values:
        p = (values==v).sum()/size
        value += p ** 2
    return 1- value


def gini(values, cond_values):
    """加权gini指数 """
    cond_gini = 0
    size = len(values)
    for cond in np.unique(cond_values):
        mask = cond_values==cond
        cond_gini += mask.sum()/size * feat_gini(values[mask])
    return cond_gini


def is_discrete(feature):
    """判断特征是否离散取值"""
    return isinstance(feature.dtype, pd.CategoricalDtype) or is_string_dtype(feature) or is_bool(feature)


def is_samples_same(data):
    """判断数据集中样本取值是否全部相同"""
    same_feats = (data.nunique(axis=0) == 1).unique()
    return len(same_feats) == 1 and sum(same_feats) == 1


def draw_tree(d_tree, f_name='tree_fig'):
    g = graphviz.Digraph()
    def draw(node, parent=None, edge_label=None):
        if parent is None:
            g.node(node.id, node.to_str())
        else:
            g.node(node.id, node.to_str(), penwidth='2' if node.is_leaf else '1', color='blue' if node.is_leaf else None)
            g.edge(parent, node.id, edge_label)
        if not node.is_leaf:
            for l, n in node.get_children():
                draw(n, node.id, l)
    draw(d_tree)
    g.view(f_name, cleanup=True)
