from copy import deepcopy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score

# 多分类正确率
def multiclass_acc(preds, truths):
    # round() 方法返回浮点数x的四舍五入值
    '''
    n是保留的小数点位数
    当参数n不存在时，round()
    函数的输出为整数。

    当参数n存在时，即使为0，round()函数的输出也会是一个浮点数。
    1、np.round(preds) == np.round(truths) 预测值是否等于真实值 相同为1 不同为0
    2、np.sum(np.round(preds) == np.round(truths)) 求和 算出所有为1的总数 也就是预测正确的个事
    3、np.sum(np.round(preds) == np.round(truths)) / float(len(truths)) 预测正确的个数除以总个数 算出正确率
    '''
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

# 计算正确率权重
def weighted_acc(preds, truths, verbose):
    preds = preds.view(-1)
    truths = truths.view(-1)

    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1

    w_acc = (tp * n / p + tn) / (2 * n)

    if verbose:
        fp = n - tn
        fn = p - tp
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        # 预测结果中正确的正面 正确的负面 错误的正面 错误的负面 正面 负面 召回率 精确率
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n, 'Recall', recall, "f1", f1)

    return w_acc

# 评估MOSEI情绪 他是一个根据回归值区间划分的分类任务
# sentiment 情绪
def eval_mosei_senti(results, truths, exclude_zero=False):
    # 看成1列
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    # 7分类
    # clip修剪
    # 函数功能：把数组里面的数压缩到设定的值范围,不是放缩，而是截断
    # 大于a_max的值都设置为a_max,小于a_min的值都设置为a_min,介于a_min和a_max之间的值保留不动
    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)

    # 5分类
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    # preds和真相之间的平均L1距离 绝对值
    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    # 计算两组数据的相关系数
    #https://blog.csdn.net/qq39514033/article/details/88931639
    corr = np.corrcoef(test_preds, test_truth)[0][1]

    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    acc5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f1 = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

    # 二分类
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    #https://blog.csdn.net/u011630575/article/details/79645814
    acc2 = accuracy_score(binary_truth, binary_preds)

    return mae, acc2, acc5, acc7, f1, corr

def eval_mosei_emo(preds, truths, threshold, verbose=False):
    '''
    CMU-MOSEI Emotion is a multi-label classification task
    CMU-MOSEI 情感是一项多标签分类任务 一个片段可能有许多情感 不只是一个
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    total = preds.size(0)
    num_emo = preds.size(1)

    # tensor转cpu 去掉梯度
    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    # 经过sigmoid激活函数
    preds = torch.sigmoid(preds)

    # https://www.cnblogs.com/yjybupt/p/12930869.html
    # roc_auc_score受试者工作特性曲线
    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))

    preds[preds > threshold] = 1
    preds[preds <= threshold] = 0

    accs = []
    f1s = []
    for emo_ind in range(num_emo):
        preds_i = preds[:, emo_ind]
        truths_i = truths[:, emo_ind]
        accs.append(weighted_acc(preds_i, truths_i, verbose=verbose))
        f1s.append(f1_score(truths_i, preds_i, average='weighted'))

    accs.append(np.average(accs))
    f1s.append(np.average(f1s))

    acc_strict = 0
    acc_intersect = 0
    acc_subset = 0
    for i in range(total):
        if torch.all(preds[i] == truths[i]):
            acc_strict += 1
            acc_intersect += 1
            acc_subset += 1
        else:
            is_loose = False
            is_subset = False
            for j in range(num_emo):
                if preds[i, j] == 1 and truths[i, j] == 1:
                    is_subset = True
                    is_loose = True
                elif preds[i, j] == 1 and truths[i, j] == 0:
                    is_subset = False
                    break
            if is_subset:
                acc_subset += 1
            if is_loose:
                acc_intersect += 1

    acc_strict /= total # all correct 全部正确
    acc_intersect /= total # at least one emotion is predicted 至少有一种情绪是可以预测的
    acc_subset /= total # predicted is a subset of truth 预测是真的子集

    return accs, f1s, aucs, [acc_strict, acc_subset, acc_intersect]


def eval_iemocap(preds, truths, best_thresholds=None):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    # 情感列表=["高兴"，"伤心"，"生气"，"中性"] 四种情感
    '''
    preds: (bs, num_emotions)
    truths: (bs, num_emotions)
    '''

    num_emo = preds.size(1)

    preds = preds.cpu().detach()
    truths = truths.cpu().detach()

    preds = torch.sigmoid(preds)

    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))

    if best_thresholds is None:
        # select the best threshold for each emotion category, based on F1 score
        # 根据F1得分，为每个情绪类别选择最佳阈值
        thresholds = np.arange(0.05, 1, 0.05)
        _f1s = []
        for t in thresholds:
            _preds = deepcopy(preds)
            _preds[_preds > t] = 1
            _preds[_preds <= t] = 0

            this_f1s = []

            for i in range(num_emo):
                pred_i = _preds[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i))

            _f1s.append(this_f1s)
        _f1s = np.array(_f1s)
        best_thresholds = (np.argmax(_f1s, axis=0) + 1) * 0.05

    # th = [0.5] * truths.size(1)
    for i in range(num_emo):
        pred = preds[:, i]
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds[:, i] = pred

    accs = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds[:, i]
        truth_i = truths[:, i]

        acc = weighted_acc(pred_i, truth_i, verbose=False)
        # acc = accuracy_score(truth_i, pred_i)
        recall = recall_score(truth_i, pred_i)
        precision = precision_score(truth_i, pred_i)
        f1 = f1_score(truth_i, pred_i)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    accs.append(np.average(accs))
    recalls.append(np.average(recalls))
    precisions.append(np.average(precisions))
    f1s.append(np.average(f1s))

    return (accs, recalls, precisions, f1s, aucs), best_thresholds

def eval_iemocap_ce(preds, truths):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    '''
    preds: (num_of_data, 4)
    truths: (num_of_data,)
    '''
    preds = preds.argmax(-1)
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro')
    r = recall_score(truths, preds, average='macro')
    p = precision_score(truths, preds, average='macro')
    return acc, r, p, f1
