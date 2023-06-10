# -*- coding: utf-8 -*
# @Time: 2023-04-02 13:20
# @Author: Three
# @File: train.py
# Software: PyCharm


import torch
from termcolor import colored
from transformers import AutoTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
import transformers
from datasets import load_dataset, load_metric, DatasetDict, concatenate_datasets, Dataset
import torch.utils.data as Data
import datasets
import random
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel
import evaluate
import numpy as np
import tensorflow as tf
import os
from evaluateMetric.glue.glue import Glue
import torch.nn as nn
from capsulnet import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import time
from torch.nn.utils.weight_norm import weight_norm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("start")
print("build model")
device = torch.device("cuda", 0)
# device = torch.device("cpu")

print(colored(f"{transformers.__version__}", "blue"))
# send_example_telemetry("text_classification_notebook", framework="pytorch")
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task = "cola"

model_checkpoint_1024 = "models/SpliceBERT.1024nt"
model_checkpoint_510 = "models/SpliceBERT-human.510nt"

model_type = 'integrate_final'
batch_size = 8

actual_task = "mnli" if task == "mnli-mm" else task
# metric = evaluate.load("glue", config_name=task)
metric = Glue(config_name=task)
print("metric:\n", metric)

# load positive and negative data
# train_dataset_0 = DatasetDict.from_csv({'train': '/mnt/sdb/home/lsr/RNAmodofication/data/process/train/0.csv'})
# train_dataset_1 = DatasetDict.from_csv({'train': '/mnt/sdb/home/lsr/RNAmodofication/data/process/train/1.csv'})

# print(colored(f"dataset:{dataset}", "blue"))
# print("dataset[train][0]:", dataset["train"][0])
"""
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'data', 'label'],
        num_rows: 2781
    })
    test: Dataset({
        features: ['Unnamed: 0', 'data', 'label'],
        num_rows: 100
    })
    valid: Dataset({
        features: ['Unnamed: 0', 'data', 'label'],
        num_rows: 300
    })
})

dataset[train][0]: {'idx': 0, 
'data': ' C C C C A T A C C C C T A G T C A T C C T C G G C A G G T C T C A G T C C C G G C T C C A T C T G T G C C C T C G C C C C A G C C G C A G C T A T G T T G C A C A C C G A G G G C C A C G C T C T T C T T C G G G C G G T G G G T C A G G G T A A G C T A C G C T T G G C C C G T T T G C T T C T G G A G G G A G G C G C C T A C G T G A A T G A G G G T G A T G C G C A G G G G G A G A C T G C G C T A A T G G C A G C C T G T C G G G C C C G C T A C G A C G A C C C C C A G A A C A A G G C A C G C A T G G T A C G C T A C C T C C T G G A G C A A G G C G C G G A C C C C A A T A T C G C A G A C C G A T T A G G G C G C A C G G C G C T C A T G C A C G C T T G C G C C G G G G G T G G G G G C G C C G C G G T G G C C T C G C T G C T C C T T G C C C A C G G C G C A G A C C C C T C A G T C C G A G A T C A C G C G G G C G C C T C G G C T C T T G T C C A C G C C C T G G A C C G C G G G G A C C G C G A G A C C C T T G C C A C A C T G C T G G A C G C C T G C A A G G C C A A G G G T A C G G A G G T C A T C A T C A T C A C C A C C G A T A C C T C G C C C T C A G G C A C C A A G A A G A C C C G G C A G T A T C T C A A T T C T C C A C C A T C C C C A G G G G T G G A G G A C C C T G C T C C C G C C T C T C C T A G C C C G G G G T T C T G C A C G T C G C C T T C G G A A A T C C A A C T G C A G A C C G C T G G A G G A G G A G G G C G T G G G A T G T T A T C C C C T C G C G C C C A G G A A G A A G A G G A G A A G C G G G A C G T A T T T G A A T T C C C T C T T C C T A A G C C C C C C G A T G A C C C A T C C C C T T C C G A G C C G C T C C C C A A A C C A C C A C G C C A T C C C C C A A A A C C A C T C A A A A G G C T C A A C T C C G A G C C C T G G G G C C T A G T G G C C C C T C C T C A A C C A G T C C C A C C C A C T G A A G G G A G A C C G G G G A T C G A G C G C T T G A C T G C C G A A T T C A A T G G C C T G A C C C T G A C C G G T C G A C C C C G T C T T T C C C G A C G T C A C A G C A C C G A A G G C C C T G A G G A C C C G C C C C C A T G G G C G G A G A A A G T G A C T A G C G G G G G T C C T', 
'label': 0}
"""


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        # dims : 128 384
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]    # 128
            out_dim = dims[i + 1]   # 384
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k  # 3
        self.v_dim = v_dim  # 128
        self.q_dim = q_dim  # 128
        self.h_dim = h_dim  # 128
        self.h_out = h_out  # 2

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)   # drug_length
        q_num = q.size(1)   # protein_length
        if self.h_out <= self.c:    # multi-head
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0



class BertClassificationModel(nn.Module):
    def __init__(self, hidden_size=128, device='cuda'):
        super(BertClassificationModel, self).__init__()

        self.device = device
        # # 读取分词器
        self.tokenizer_510 = AutoTokenizer.from_pretrained(model_checkpoint_510, use_fast=True)
        self.tokenizer_1024 = AutoTokenizer.from_pretrained(model_checkpoint_1024, use_fast=True)

        # 读取预训练模型
        self.bert_510 = AutoModel.from_pretrained(pretrained_model_name_or_path=model_checkpoint_510).to(device)
        self.bert_1024 = AutoModel.from_pretrained(pretrained_model_name_or_path=model_checkpoint_1024).to(device)


        self.bcn = weight_norm(BANLayer(v_dim=512, q_dim=512, h_dim=512, h_out=2), name='h_mat', dim=None)

        # linear
        self.block = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )
        # self.fc = nn.Linear(hidden_size, 2)
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)

    def forward(self, batch_sentences):  # [batch_size,1]
        batch_sentences = list(batch_sentences)
        batch_sentences_partial = [seq[491:1511] for seq in batch_sentences]
        token_seq = self.tokenizer_510(batch_sentences_partial,
                                   truncation=True,
                                   return_tensors="pt", max_length=512)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        representation_1 = self.bert_510(input_ids=input_ids.to(self.device), token_type_ids=token_type_ids.to(self.device),
                                   attention_mask=attention_mask.to(self.device))['last_hidden_state']


        token_seq = self.tokenizer_1024(batch_sentences,
                                   truncation=True,
                                   return_tensors="pt")
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        representation_2 = self.bert_1024(input_ids=input_ids.to(self.device), token_type_ids=token_type_ids.to(self.device),
                                   attention_mask=attention_mask.to(self.device))['last_hidden_state']

        f, att = self.bcn(representation_1, representation_2)

        pred = self.block(f)

        return pred


# show some data in datasets randomly
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    # display(HTML(df.to_html()))
    print(colored(df, "blue"))

class RNADataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label)
    def __getitem__(self, i):
        return self.data[i], self.label[i]
    def __len__(self):
        return len(self.label)

def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
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

    if (tp + fp) == 0:
        PRE = 0
    else:
        PRE = float(tp) / (tp + fp)

    BACC = 0.5 * Sensitivity + 0.5 * Specificity

    performance = [ACC, BACC, Sensitivity, Specificity, MCC, AUC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def evaluate(data_iter, net, criterion):
    pred_prob = []
    label_pred = []
    label_real = []

    for j, (data, labels) in enumerate(data_iter, 0):

        labels = labels.to(device)
        output = net(data)
        loss = criterion(output, labels)

        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data, loss

def to_log(log, index):
    with open(f"./results/{index + 1}/result_{model_type}.log", "a+") as f:
        f.write(log + '\n')

# show_random_elements(dataset["train"])
"""
show_random_elements(dataset["train"])  output:
    idx                                               data  label
0  2245   G T T T C C A G A C C A T G C C G A G C C A C...      1
1  2730   A G A A C C T C G A G G T C C G C G C G C G C...      1
2   561   A G T C A T G G G A G A G G G G T C A C A T G...      0
3  2595   C G A T G A C T G T G T G G T T G C C C A A A...      1
4  1238   A C T C C G T A T T T C A C C T G A A C A G T...      0
5  2156   G T T C A T C C A A G T A A C G A T T C T C C...      1
6  2450   A G C T G T G G G T A G T G T T C T C T C A T...      1
7  2151   G T G G A A G G A T T A C T T T A G C C C A G...      1
8   449   G C A C G C A G T G G C G T G A T C T C T G C...      0
9  2195   C A G T G G C C T G A T C A T C A G T C T T G...      1
"""

#  你可以直接用你的预测和标签调用它的计算方法，它会返回一个带有度量值的字典：
# fake_preds = np.random.randint(0, 2, size=(64,))
# fake_labels = np.random.randint(0, 2, size=(64,))
# metric.compute(predictions=fake_preds, references=fake_labels)


# print(tokenizer(" C C C C A T A C C C C T A G T C A T C C T C G G C A G G T C T C A G T C C C G G C T C C A T C T G T G C C C T C G C C C C A G C C G C A G C T A T G T T G C A C A C C G A G G G C C A C G C T C T T C T T C G G G C G G T G G G T C A G G G T A A G C T A C G C T T G G C C C G T T T G C T T C T G G A G G G A G G C G C C T A C G T G A A T G A G G G T G A T G C G C A G G G G G A G A C T G C G C T A A T G G C A G C C T G T C G G G C C C G C T A C G A C G A C C C C C A G A A C A A G G C A C G C A T G G T A C G C T A C C T C C T G G A G C A A G G C G C G G A C C C C A A T A T C G C A G A C C G A T T A G G G C G C A C G G C G C T C A T G C A C G C T T G C G C C G G G G G T G G G G G C G C C G C G G T G G C C T C G C T G C T C C T T G C C C A C G G C G C A G A C C C C T C A G T C C G A G A T C A C G C G G G C G C C T C G G C T C T T G T C C A C G C C C T G G A C C G C G G G G A C C G C G A G A C C C T T G C C A C A C T G C T G G A C G C C T G C A A G G C C A A G G G T A C G G A G G T C A T C A T C A T C A C C A C C G A T A C C T C G C C C T C A G G C A C C A A G A A G A C C C G G C A G T A T C T C A A T T C T C C A C C A T C C C C A G G G G T G G A G G A C C C T G C T C C C G C C T C T C C T A G C C C G G G G T T C T G C A C G T C G C C T T C G G A A A T C C A A C T G C A G A C C G C T G G A G G A G G A G G G C G T G G G A T G T T A T C C C C T C G C G C C C A G G A A G A A G A G G A G A A G C G G G A C G T A T T T G A A T T C C C T C T T C C T A A G C C C C C C G A T G A C C C A T C C C C T T C C G A G C C G C T C C C C A A A C C A C C A C G C C A T C C C C C A A A A C C A C T C A A A A G G C T C A A C T C C G A G C C C T G G G G C C T A G T G G C C C C T C C T C A A C C A G T C C C A C C C A C T G A A G G G A G A C C G G G G A T C G A G C G C T T G A C T G C C G A A T T C A A T G G C C T G A C C C T G A C C G G T C G A C C C C G T C T T T C C C G A C G T C A C A G C A C C G A A G G C C C T G A G G A C C C G C C C C C A T G G G C G G A G A A A G T G A C T A G C G G G G G T C C T"))
"""
tokenizer output：
'input_ids': [0, 230, 230, 230, 230, 83, 255, 83, 230, 230, 230, 230, 255, 83, 272, 255, 230, 83, 255, 230, 230, 255, 230, 272, 272, 230, 83, 272, 272, 255, 230, 255, 230, 83, 272, 255, 230, 230, 230, 272, 272, 230, 255, 230, 230, 83, 255, 230, 255, 272, 255, 272, 230, 230, 230, 255, 230, 272, 230, 230, 230, 230, 83, 272, 230, 230, 272, 230, 83, 272, 230, 255, 83, 255, 272, 255, 255, 272, 230, 83, 230, 83, 230, 230, 272, 83, 272, 272, 272, 230, 230, 83, 230, 272, 230, 255, 230, 255, 255, 230, 255, 255, 230, 272, 272, 272, 230, 272, 272, 255, 272, 272, 272, 255, 230, 83, 272, 272, 272, 255, 83, 83, 272, 230, 255, 83, 230, 272, 230, 255, 255, 272, 272, 230, 230, 230, 272, 255, 255, 255, 272, 230, 255, 255, 230, 255, 272, 272, 83, 272, 272, 272, 83, 272, 272, 230, 272, 230, 230, 255, 83, 230, 272, 255, 272, 83, 83, 255, 272, 83, 272, 272, 272, 255, 272, 83, 255, 272, 230, 272, 230, 83, 272, 272, 272, 272, 272, 83, 272, 83, 230, 255, 272, 230, 272, 230, 255, 83, 83, 255, 272, 272, 230, 83, 272, 230, 230, 255, 272, 255, 230, 272, 272, 272, 230, 230, 230, 272, 230, 255, 83, 230, 272, 83, 230, 272, 83, 230, 230, 230, 230, 230, 83, 272, 83, 83, 230, 83, 83, 272, 272, 230, 83, 230, 272, 230, 83, 255, 272, 272, 255, 83, 230, 272, 230, 255, 83, 230, 230, 255, 230, 230, 255, 272, 272, 83, 272, 230, 83, 83, 272, 272, 230, 272, 230, 272, 272, 83, 230, 230, 230, 230, 83, 83, 255, 83, 255, 230, 272, 230, 83, 272, 83, 230, 230, 272, 83, 255, 255, 83, 272, 272, 272, 230, 272, 230, 83, 230, 272, 272, 230, 272, 230, 255, 230, 83, 255, 272, 230, 83, 230, 272, 230, 255, 255, 272, 230, 272, 230, 230, 272, 272, 272, 272, 272, 255, 272, 272, 272, 272, 272, 230, 272, 230, 230, 272, 230, 272, 272, 255, 272, 272, 230, 230, 255, 230, 272, 230, 255, 272, 230, 255, 230, 230, 255, 255, 272, 230, 230, 230, 83, 230, 272, 272, 230, 272, 230, 83, 272, 83, 230, 230, 230, 230, 255, 230, 83, 272, 255, 230, 230, 272, 83, 272, 83, 255, 230, 83, 230, 272, 230, 272, 272, 272, 230, 272, 230, 230, 255, 230, 272, 272, 230, 255, 230, 255, 255, 272, 255, 230, 230, 83, 230, 272, 230, 230, 230, 255, 272, 272, 83, 230, 230, 272, 230, 272, 272, 272, 272, 83, 230, 230, 272, 230, 272, 83, 272, 83, 230, 230, 230, 255, 255, 272, 230, 230, 83, 230, 83, 230, 255, 272, 230, 255, 272, 272, 83, 230, 272, 230, 230, 255, 272, 230, 83, 83, 272, 272, 230, 230, 83, 83, 272, 272, 272, 255, 83, 230, 272, 272, 83, 272, 272, 255, 230, 83, 255, 230, 83, 255, 230, 83, 255, 230, 83, 230, 230, 83, 230, 230, 272, 83, 255, 83, 230, 230, 255, 230, 272, 230, 230, 230, 255, 230, 83, 272, 272, 230, 83, 230, 230, 83, 83, 272, 83, 83, 272, 83, 230, 230, 230, 272, 272, 230, 83, 272, 255, 83, 255, 230, 255, 230, 83, 83, 255, 255, 230, 255, 230, 230, 83, 230, 230, 83, 255, 230, 230, 230, 230, 83, 272, 272, 272, 272, 255, 272, 272, 83, 272, 272, 83, 230, 230, 230, 255, 272, 230, 255, 230, 230, 230, 272, 230, 230, 255, 230, 255, 230, 230, 255, 83, 272, 230, 230, 230, 272, 272, 272, 272, 255, 255, 230, 255, 272, 230, 83, 230, 272, 255, 230, 272, 230, 230, 255, 255, 230, 272, 272, 83, 83, 83, 255, 230, 230, 83, 83, 230, 255, 272, 230, 83, 272, 83, 230, 230, 272, 230, 255, 272, 272, 83, 272, 272, 83, 272, 272, 83, 272, 272, 272, 230, 272, 255, 272, 272, 272, 83, 255, 272, 255, 255, 83, 255, 230, 230, 230, 230, 255, 230, 272, 230, 272, 230, 230, 230, 83, 272, 272, 83, 83, 272, 83, 83, 272, 83, 272, 272, 83, 272, 83, 83, 272, 230, 272, 272, 272, 83, 230, 272, 255, 83, 255, 255, 255, 272, 83, 83, 255, 255, 230, 230, 230, 255, 230, 255, 255, 230, 230, 255, 83, 83, 272, 230, 230, 230, 230, 230, 230, 272, 83, 255, 272, 83, 230, 230, 230, 83, 255, 230, 230, 230, 230, 255, 255, 230, 230, 272, 83, 272, 230, 230, 272, 230, 255, 230, 230, 230, 230, 83, 83, 83, 230, 230, 83, 230, 230, 83, 230, 272, 230, 230, 83, 255, 230, 230, 230, 230, 230, 83, 83, 83, 83, 230, 230, 83, 230, 255, 230, 83, 83, 83, 83, 272, 272, 230, 255, 230, 83, 83, 230, 255, 230, 230, 272, 83, 272, 230, 230, 230, 255, 272, 272, 272, 272, 230, 230, 255, 83, 272, 255, 272, 272, 230, 230, 230, 230, 255, 230, 230, 255, 230, 83, 83, 230, 230, 83, 272, 255, 230, 230, 230, 83, 230, 230, 230, 83, 230, 255, 272, 83, 83, 272, 272, 272, 83, 272, 83, 230, 230, 272, 272, 272, 272, 83, 255, 230, 272, 83, 272, 230, 272, 230, 255, 255, 272, 83, 230, 255, 272, 230, 230, 272, 83, 83, 255, 255, 230, 83, 83, 255, 272, 272, 230, 230, 255, 272, 83, 230, 230, 230, 255, 272, 83, 230, 230, 272, 272, 255, 230, 272, 83, 230, 230, 230, 230, 272, 255, 230, 255, 255, 255, 230, 230, 230, 272, 83, 230, 272, 255, 230, 83, 230, 83, 272, 230, 83, 230, 230, 272, 83, 83, 272, 272, 230, 230, 230, 255, 272, 83, 272, 272, 83, 230, 230, 230, 272, 230, 230, 230, 230, 230, 83, 255, 272, 272, 272, 230, 272, 272, 83, 272, 83, 83, 83, 272, 255, 272, 83, 230, 255, 83, 272, 230, 272, 272, 272, 272, 272, 255, 230, 230, 255, 2]
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
tokenizer successfully 
C: 230
A: 83
T: 255
G: 272
"""

# Note that load_metric has loaded the proper metric associated to your task, which is:
# 注意其中的映射结果要改成相应的类名，我们选择了cola因此我只更改了cola对应的列名
task_to_keys = {
    "cola": ("data", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

num_labels = 2
metric_name = "accuracy"

def main():
    batchsize = 8  # 定义每次放多少个数据参加训练

    for index in range(5,11):

        train_dataset_0 = DatasetDict.from_csv({'train': f'data/reprocess/train/{2*index}.csv'})
        train_dataset_1 = DatasetDict.from_csv({'train': f'data/reprocess/train/{2*index+1}.csv'})
        train_dataset = concatenate_datasets([train_dataset_0['train'], train_dataset_1['train']])

        valid_dataset_0 = DatasetDict.from_csv({'valid': f'data/reprocess/valid/{2*index}.csv'})
        valid_dataset_1 = DatasetDict.from_csv({'valid': f'data/reprocess/valid/{2*index+1}.csv'})
        valid_dataset = concatenate_datasets([valid_dataset_0['valid'], valid_dataset_1['valid']])

        test_dataset_0 = DatasetDict.from_csv({'test': f'data/reprocess/test/{2*index}.csv'})
        test_dataset_1 = DatasetDict.from_csv({'test': f'data/reprocess/test/{2*index+1}.csv'})

        test_dataset = concatenate_datasets([test_dataset_0['test'], test_dataset_1['test']])

        dataset = DatasetDict()
        dataset['train'] = train_dataset
        dataset['test'] = test_dataset
        dataset['valid'] = valid_dataset

        trainDatas = RNADataset(train_dataset['data'], train_dataset['label'])
        validDatas = RNADataset(valid_dataset['data'], valid_dataset['label'])
        testDatas = RNADataset(test_dataset['data'], test_dataset['label'])

        train_loader = Data.DataLoader(trainDatas, batch_size=batchsize, shuffle=True)
        valid_loader = Data.DataLoader(validDatas, batch_size=batchsize, shuffle=True)
        test_loader = Data.DataLoader(testDatas, batch_size=batchsize, shuffle=True)

        # 这里搭建训练循环，输出训练结果

        epoch_num = 50  # 设置循环多少次训练，可根据模型计算情况做调整，如果模型陷入了局部最优，那么循环多少次也没啥用

        print('training...')

        # 初始化模型
        model = BertClassificationModel(device=device).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # 首先定义优化器，这里用的AdamW，lr是学习率，因为bert用的就是这个

        # 这里是定义损失函数，交叉熵损失函数比较常用解决分类问题
        # 依据你解决什么问题，选择什么样的损失函数
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping()
        # criterion = MarginLoss(0.9, 0.1, 0.5)
        best_acc = 0
        print("模型数据已经加载完成,现在开始模型训练。")

        for epoch in range(epoch_num):
            loss_ls = []
            t0 = time.time()
            model.train()
            for i, (data, labels) in enumerate(train_loader):
                # print(len(data))
                if len(data) <= 1:
                    continue
                # labels = data['label']
                labels = labels.to(device)
                output = model(data)
                optimizer.zero_grad()  # 梯度清0
                loss = criterion(output, labels)  # 计算误差
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                loss_ls.append(loss.item())

                # 打印一下每一次数据扔进去学习的进展
                # print('batch:%d loss:%.5f' % (i, loss.item()))

            # 打印一下每个epoch的深度学习的进展i
            # print('epoch:%d loss:%.5f' % (epoch, loss.item()))

        # 下面开始测试模型是不是好用哈
            print('testing...(约2000秒(CPU))')

            model.eval()
            with torch.no_grad():
                train_performance, train_roc_data, train_prc_data, _ = evaluate(train_loader, model, criterion)
                valid_performance, valid_roc_data, valid_prc_data, valid_loss = evaluate(valid_loader, model, criterion)

            # results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
            results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
            results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
            results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                       + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
                valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
            print(results)
            to_log(results, index)
            valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
            if valid_acc > best_acc:
                best_acc = valid_acc
                test_performance, test_roc_data, test_prc_data, _ = evaluate(test_loader, model, criterion)
                # best_performance = valid_performance
                filename = '{}, {}[{:.4f}].pt'.format(f'model{model_type}' + ', epoch[{}]'.format(epoch + 1), 'ACC', test_performance[0])
                save_path_pt = os.path.join(f'Saved_Models/{index+1}', filename)
                torch.save(model.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
                test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
                    epoch + 1) + '=' * 16 \
                               + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tAUC,\tPRE]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                    test_performance[4], test_performance[5]) + '\n' + '=' * 60
                print(test_results)
                to_log(test_results, index)
                test_ROC = valid_roc_data
                test_PRC = valid_prc_data
                save_path_roc = os.path.join(f'ROC/{index+1}', filename)
                save_path_prc = os.path.join(f'PRC/{index+1}', filename)
                torch.save(test_roc_data, save_path_roc, _use_new_zipfile_serialization=False)
                torch.save(test_prc_data, save_path_prc, _use_new_zipfile_serialization=False)

            early_stopping(valid_acc, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break


if __name__ == '__main__':
    main()


