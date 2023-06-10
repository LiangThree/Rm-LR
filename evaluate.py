# -*- coding: utf-8 -*
# @Time: 2023-03-25 16:29
# @Author: Three
# @File: evaluate.py
# Software: PyCharm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score


def evaluate(data_iter, net, device):  # data_iter是输入的数据， net是构建的ExamPle网络
    pred_prob = []
    label_pred = []
    label_real = []
    for rna_seq, label in data_iter:
        rna_seq, label = rna_seq.to(device), label.to(device)
        outputs = net(rna_seq)
        outputs_cpu = outputs.cpu()
        y_cpu = label.cpu()
        pred_prob_positive = outputs_cpu[:, 1]  # [:, 1]表示第二列
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0  # 结果是1且预测正确
    fp = 0  # 结果是1且预测错误
    tn = 0  # 结果是0且预测正确
    fn = 0  # 结果是0且预测错误
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    # PRE = float(tp) / (tp + fp)
    PRE = float(tp) / (tp + fp + 1)  # todo 为了防止除0错误，加了1

    performance = [ACC, Sensitivity, Specificity, AUC, PRE, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

