"""
使用PyTorch实现的单个GRU单元的代码
结合两种信息
"""
from termcolor import colored

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
import pandas as pd
import csv

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H.%M')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda", 0)
print("Begin...")


class GRUModel(nn.Module):
    def __init__(self, vocab_size=24, emb_dim=100, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.2)

        self.block = nn.Sequential(
            nn.Linear(6784, 1024),  # 1001:128384  51:6784  510：65536
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
        output = x.permute(1, 0, 2)
        output, hn = self.gru(output)
        output = output.permute(1, 0, 2)  # [4019, 52, 100]
        hn = hn.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)
        hn = hn.reshape(output.shape[0], -1)
        output = torch.cat([output, hn], 1)  # [4019, 6912]
        #         print(output.shape,hn.shape)
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

# result_dir = "GRUModel_ExamPle_Result_raw"
result_dir = "Result/GRUModel_ExamPle_Result_raw_51"


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
    """
    删除文件夹下的文件，保留但清空子文件夹
    :param filepath:
    :return:
    """
    for i in os.listdir(filepath):
        # 如果存在文件夹进行递归
        if os.path.isdir(os.path.join(filepath, i)):
            del_file(os.path.join(filepath, i))
        # 如果是文件进行删除
        elif os.path.isfile:
            os.remove(os.path.join(filepath, i))


# del_file(f"./{result_dir}")
# print("删除先前运行结果")
# exit()

for data_num in range(0, 10):
    print(f"current:{data_num}")
    train_iter, test_iter, valid_iter = DataLoad(f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/train/{2*data_num}.csv',
                                                 f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/train/{2*data_num+1}.csv',
                                                 f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/test/{2*data_num}.csv',
                                                 f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/test/{2*data_num+1}.csv',
                                                 f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/valid/{2*data_num}.csv',
                                                 f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/valid/{2*data_num+1}.csv')

    print("model start...")

    net = GRUModel().to(device)
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


