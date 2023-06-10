import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch
from termcolor import colored
from torch.autograd import Variable

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import os
import torch
import torch.nn as nn
import time
import datetime
from data.dataModule import DataLoad
from evaluate import evaluate
import csv

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H.%M')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda", 0)
print("Begin...")

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=24, emb_dim=100, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        layer_num = 2
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=layer_num)
        self.h0 = torch.randn(layer_num, 510, hidden_dim)  # 初始隐藏状态，层数为2，批次大小为3，隐藏维度为20
        self.c0 = torch.randn(layer_num, 510, hidden_dim)  # 初始细胞状态
        self.dropout = nn.Dropout(p=0.3)

        self.block = nn.Sequential(
            nn.Linear(32640, 1024),  # 40:2560  1001:64064  51:3264 510:32640
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, x):
        x = x.to('cuda')
        x = self.embedding(x)  # [4019, 52] -> [4019, 52, 100]
        x = self.dropout(x)
        self.h0 = self.h0.to('cuda')
        self.c0 = self.c0.to('cuda')
        output, (hn, cn) = self.lstm(x, (self.h0, self.c0))  # 输出[4019, 52, 64]

        output = output.reshape(output.shape[0], -1)  # [4019, 3328]
        #  print(output.shape,hn.shape)
        return self.block(output)

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

result_dir = "LSTM_ExamPle_Result_510"


def to_log(log):
    with open(f"./{result_dir}/Log/Log_dataset:{data_num}_{now}.log", "a+") as f:
        f.write(log + '\n')


import csv
def insert_csv(listA):
    f = open(f"./{result_dir}/Log.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(listA)
    f.close()

def del_file(filepath):
    for i in os.listdir(filepath):
        # 如果存在文件夹进行递归
        if os.path.isdir(os.path.join(filepath, i)):
            del_file(os.path.join(filepath, i))
        # 如果是文件进行删除
        elif os.path.isfile:
            os.remove(os.path.join(filepath, i))


# del_file(f"./{result_dir}")
# exit(0)

for data_num in range(0, 10):
    print(f"current:{data_num}")
    train_iter, test_iter, valid_iter = DataLoad(
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/train/{2 * data_num}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/train/{2 * data_num + 1}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/test/{2 * data_num}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/test/{2 * data_num + 1}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/valid/{2 * data_num}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/valid/{2 * data_num + 1}.csv')

    print("model start...")


    net = LSTMModel().to(device)
    lr = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion_model = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping()

    best_acc = 0
    EPOCH = 50
    for epoch in range(EPOCH):
        # for epoch in tqdm(range(EPOCH)):
        loss_ls = []
        t0 = time.time()
        net.train()
        for rna_seq, label in train_iter:
            rna_seq, label = rna_seq.to(device), label.to(device)
            print(rna_seq, label)
            exit(0)
            pred = net(rna_seq)


            loss = criterion_model(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())


        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data = evaluate(train_iter, net, device)
            valid_performance, valid_roc_data, valid_prc_data = evaluate(valid_iter, net, device)

        #  results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tPRE,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        print(results)
        to_log(results)
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        if valid_acc > best_acc:
            test_performance, test_roc_data, test_prc_data = evaluate(test_iter, net, device)
            best_acc = valid_acc
            best_performance = test_performance
            # save_path_pt = os.path.join('./TransformerEncoder_Result/Model', '{}, {}[{:.3f}].pt'.format('epoch[{}]'.format(epoch + 1), 'ACC', best_acc))
            # filename = f'epoch[{epoch + 1}], ACC[{"%.2f"%best_acc}]'

            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tPRE,\t\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                best_performance[4], best_performance[5]) + '\n' + '=' * 60
            print(best_results)
            to_log(best_results)
            best_performance.insert(0, epoch + 1)
            best_performance.insert(0, data_num)
            insert_csv(best_performance)
            best_ROC = test_roc_data
            best_PRC = test_prc_data

            # save_path_pt = f'./TransformerEncoder_Result/Model/epoch[{epoch + 1}], ACC[{"%.2f" % best_acc}]'
            save_path_pt = f'./{result_dir}/Model/dataset:[{data_num}]_epoch:[{epoch + 1}]_ACC:[{"%.2f" % best_acc}]'
            if not os.path.exists(save_path_pt):
                os.system(r"touch {}".format(save_path_pt))
            torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)

            save_path_roc = os.path.join(f'./{result_dir}/ROC/dataset:[{data_num}]_epoch[{epoch + 1}]_{now}')
            if not os.path.exists(save_path_roc):
                os.system(r"touch {}".format(save_path_roc))

            save_path_prc = os.path.join(f'./{result_dir}/PRC/dataset:[{data_num}]_epoch[{epoch + 1}]_{now}')
            if not os.path.exists(save_path_prc):
                os.system(r"touch {}".format(save_path_prc))
            torch.save(best_ROC, save_path_roc, _use_new_zipfile_serialization=False)
            torch.save(best_PRC, save_path_prc, _use_new_zipfile_serialization=False)

        early_stopping(valid_acc, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break


    print('Finished Training...')


# def sigmoid(x):
#     return 1. / (1 + np.exp(-x))
#
#
# def sigmoid_derivative(values):
#     return values * (1 - values)
#
#
# def tanh_derivative(values):
#     return 1. - values ** 2
#
#
# # createst uniform random array w/ values in [a,b) and shape args
# def rand_arr(a, b, *args):
#     np.random.seed(0)
#     return np.random.rand(*args) * (b - a) + a
#
#
# class LstmParam:
#     def __init__(self, mem_cell_ct, x_dim):
#         self.mem_cell_ct = mem_cell_ct
#         self.x_dim = x_dim
#         concat_len = x_dim + mem_cell_ct
#         # weight matrices
#         self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
#         self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
#         self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
#         self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
#         # bias terms
#         self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
#         self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
#         self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
#         self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
#         # diffs (derivative of loss function w.r.t. all parameters)
#         self.wg_diff = np.zeros((mem_cell_ct, concat_len))
#         self.wi_diff = np.zeros((mem_cell_ct, concat_len))
#         self.wf_diff = np.zeros((mem_cell_ct, concat_len))
#         self.wo_diff = np.zeros((mem_cell_ct, concat_len))
#         self.bg_diff = np.zeros(mem_cell_ct)
#         self.bi_diff = np.zeros(mem_cell_ct)
#         self.bf_diff = np.zeros(mem_cell_ct)
#         self.bo_diff = np.zeros(mem_cell_ct)
#
#     def apply_diff(self, lr=1):
#         self.wg -= lr * self.wg_diff
#         self.wi -= lr * self.wi_diff
#         self.wf -= lr * self.wf_diff
#         self.wo -= lr * self.wo_diff
#         self.bg -= lr * self.bg_diff
#         self.bi -= lr * self.bi_diff
#         self.bf -= lr * self.bf_diff
#         self.bo -= lr * self.bo_diff
#         # reset diffs to zero
#         self.wg_diff = np.zeros_like(self.wg)
#         self.wi_diff = np.zeros_like(self.wi)
#         self.wf_diff = np.zeros_like(self.wf)
#         self.wo_diff = np.zeros_like(self.wo)
#         self.bg_diff = np.zeros_like(self.bg)
#         self.bi_diff = np.zeros_like(self.bi)
#         self.bf_diff = np.zeros_like(self.bf)
#         self.bo_diff = np.zeros_like(self.bo)
#
#
# class LstmState:
#     def __init__(self, mem_cell_ct, x_dim):
#         self.g = np.zeros(mem_cell_ct)
#         self.i = np.zeros(mem_cell_ct)
#         self.f = np.zeros(mem_cell_ct)
#         self.o = np.zeros(mem_cell_ct)
#         self.s = np.zeros(mem_cell_ct)
#         self.h = np.zeros(mem_cell_ct)
#         self.bottom_diff_h = np.zeros_like(self.h)
#         self.bottom_diff_s = np.zeros_like(self.s)
#
#
# class LstmNode:
#     def __init__(self, lstm_param, lstm_state):
#         # store reference to parameters and to activations
#         self.state = lstm_state
#         self.param = lstm_param
#         # non-recurrent input concatenated with recurrent input
#         self.xc = None
#
#     def bottom_data_is(self, x, s_prev=None, h_prev=None):
#         # if this is the first lstm node in the network
#         if s_prev is None: s_prev = np.zeros_like(self.state.s)
#         if h_prev is None: h_prev = np.zeros_like(self.state.h)
#         # save data for use in backprop
#         self.s_prev = s_prev
#         self.h_prev = h_prev
#
#         # concatenate x(t) and h(t-1)
#         xc = np.hstack((x, h_prev))
#         self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
#         self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
#         self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
#         self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
#         self.state.s = self.state.g * self.state.i + s_prev * self.state.f
#         self.state.h = self.state.s * self.state.o
#
#         self.xc = xc
#
#     def top_diff_is(self, top_diff_h, top_diff_s):
#         # notice that top_diff_s is carried along the constant error carousel
#         ds = self.state.o * top_diff_h + top_diff_s
#         do = self.state.s * top_diff_h
#         di = self.state.g * ds
#         dg = self.state.i * ds
#         df = self.s_prev * ds
#
#         # diffs w.r.t. vector inside sigma / tanh function
#         di_input = sigmoid_derivative(self.state.i) * di
#         df_input = sigmoid_derivative(self.state.f) * df
#         do_input = sigmoid_derivative(self.state.o) * do
#         dg_input = tanh_derivative(self.state.g) * dg
#
#         # diffs w.r.t. inputs
#         self.param.wi_diff += np.outer(di_input, self.xc)
#         self.param.wf_diff += np.outer(df_input, self.xc)
#         self.param.wo_diff += np.outer(do_input, self.xc)
#         self.param.wg_diff += np.outer(dg_input, self.xc)
#         self.param.bi_diff += di_input
#         self.param.bf_diff += df_input
#         self.param.bo_diff += do_input
#         self.param.bg_diff += dg_input
#
#         # compute bottom diff
#         dxc = np.zeros_like(self.xc)
#         dxc += np.dot(self.param.wi.T, di_input)
#         dxc += np.dot(self.param.wf.T, df_input)
#         dxc += np.dot(self.param.wo.T, do_input)
#         dxc += np.dot(self.param.wg.T, dg_input)
#
#         # save bottom diffs
#         self.state.bottom_diff_s = ds * self.state.f
#         self.state.bottom_diff_h = dxc[self.param.x_dim:]
#
#
# class LstmNetwork():
#     def __init__(self, lstm_param):
#         self.lstm_param = lstm_param
#         self.lstm_node_list = []
#         # input sequence
#         self.x_list = []
#
#     def y_list_is(self, y_list, loss_layer):
#         """
#         Updates diffs by setting target sequence
#         with corresponding loss layer.
#         Will *NOT* update parameters.  To update parameters,
#         call self.lstm_param.apply_diff()
#         """
#         assert len(y_list) == len(self.x_list)
#         idx = len(self.x_list) - 1
#         # first node only gets diffs from label ...
#         loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
#         diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
#         # here s is not affecting loss due to h(t+1), hence we set equal to zero
#         diff_s = np.zeros(self.lstm_param.mem_cell_ct)
#         self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
#         idx -= 1
#
#         ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
#         ### we also propagate error along constant error carousel using diff_s
#         while idx >= 0:
#             loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
#             diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
#             diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
#             diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
#             self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
#             idx -= 1
#
#         return loss
#
#     def x_list_clear(self):
#         self.x_list = []
#
#     def x_list_add(self, x):
#         self.x_list.append(x)
#         if len(self.x_list) > len(self.lstm_node_list):
#             # need to add new lstm node, create new state mem
#             lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
#             self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))
#
#         # get index of most recent x input
#         idx = len(self.x_list) - 1
#         if idx == 0:
#             # no recurrent inputs yet
#             self.lstm_node_list[idx].bottom_data_is(x)
#         else:
#             s_prev = self.lstm_node_list[idx - 1].state.s
#             h_prev = self.lstm_node_list[idx - 1].state.h
#             self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)
