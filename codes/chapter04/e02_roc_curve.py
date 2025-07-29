import numpy as np
import matplotlib.pyplot as plt

# 设置图片字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16



def confusion_matrix(y_true, y_scores, threshold):
    """
    计算二分类的混淆矩阵。

    参数：
    - y_true: 真实类别 (0 或 1)
    - y_scores: 预测分数
    - threshold: 当前使用的分类阈值

    返回：
    - (TP, FP, FN, TN): 混淆矩阵四个值
    """
    TP = FP = FN = TN = 0
    for score, label in zip(y_scores, y_true):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 1:
            TP += 1
        elif predicted == 1 and label == 0:
            FP += 1
        elif predicted == 0 and label == 1:
            FN += 1
        else:
            TN += 1
    return TP, FP, FN, TN


def recall(TP, FN):
    """ 计算 Recall """
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0  # 避免除零


def auc(recall_list, precision_list):
    """
    使用梯形法则计算曲线下面积 (AUC)
    """
    auc = 0.0
    for i in range(len(recall_list) - 1):
        h = recall_list[i + 1] - recall_list[i]  # recall 差值
        trap_area = (precision_list[i] + precision_list[i + 1]) / 2 * h  # 梯形面积
        auc += trap_area
    return auc


def false_positive_rate(FP, TN):
    return FP / (FP + TN)


if __name__ == '__main__':
    y_scores = np.array([0.98, 0.95, 0.92, 0.90, 0.85, 0.80, 0.78, 0.75, 0.70, 0.65,
                        0.60, 0.55, 0.50, 0.45, 0.40])
    y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    # 计算不同阈值下的 Precision 和 Recall
    thresholds = sorted(y_scores, reverse=True)  # 以预测分数为阈值
    fpr_list = []
    tpr_list = []

    for threshold in thresholds:
        TP, FP, FN, TN = confusion_matrix(y_true, y_scores, threshold)
        fpr_list.append(false_positive_rate(FP, TN))
        tpr_list.append(recall(TP, FN))

    roc_auc_value = auc(fpr_list, tpr_list)
    print('ROC_AUC:', roc_auc_value)

    plt.figure()
    plt.plot(fpr_list, tpr_list, marker='o', linestyle='-', label="PR Curve")
    plt.xlabel('假正类率', fontname='SimSun')
    plt.ylabel('真正类率', fontname='SimSun')
    plt.grid()
    plt.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.13, wspace=0.2, hspace=0.1)
    plt.title(f'ROC曲线下面积 {roc_auc_value:.4}', fontname='SimSun')

    plt.show()
