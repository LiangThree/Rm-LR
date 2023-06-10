# -*- coding: utf-8 -*-
# @Time    : 2021/10/20 11:01
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: ReformerEncoder.py

import torch
import torch.nn as nn
from reformer_pytorch import ReformerLM
from termcolor import colored

import warnings

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

now = datetime.datetime.now()
now = now.strftime('%m-%d-%H.%M')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda", 0)
print("Begin...")


def get_attn_pad_mask(input_ids):
    pad_attn_mask_expand = torch.zeros_like(input_ids)
    batch_size, seq_len = input_ids.size()
    for i in range(batch_size):
        for j in range(seq_len):
            if input_ids[i][j] != 0:
                pad_attn_mask_expand[i][j] = 1

    return pad_attn_mask_expand.bool()


class ReformerEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(ReformerEncoder, self).__init__()
        vocab_size = 24
        # todo max_len = 52 ? ? ?
        max_len = 52

        # self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.reformer_encoder = ReformerLM(
            num_tokens=vocab_size,
            emb_dim=emb_dim,
            dim=hidden_dim,
            depth=2,
            heads=8,
            bucket_size=1,
            max_seq_len=max_len,
            fixed_position_emb=True,
            return_embeddings=True  # return output of last attention layer
        ).cuda()

        # self.classification = nn.Sequential(
        #                             nn.Linear(64, 32),
        #                             # nn.BatchNorm1d(256),
        #                             nn.ReLU(),
        #                             # nn.Dropout(0.5),
        #                             nn.Linear(32, 2),
        #                             )

    def forward(self, x):
        x = x.cuda()  # [4019, 52]
        padding_mask = get_attn_pad_mask(x)  # [4019, 52]
        # todo reformer_encoder 没有 Embedding
        # x = self.embedding(x)
        # representation = self.reformer_encoder(x)[:, 0, :].squeeze(1)
        # todo CUDA out of memory
        representation = self.reformer_encoder(x, input_mask=padding_mask)[:, 0, :].squeeze(1)

        # output = self.classification(representation)

        return representation


result_dir = "Reformer_Result"


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

for data_num in range(0, 12):
    print(f"current:{data_num}")
    train_iter, test_iter, valid_iter = DataLoad(
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/reprocess/train/{2 * data_num}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/reprocess/train/{2 * data_num + 1}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/reprocess/test/{2 * data_num}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/reprocess/test/{2 * data_num + 1}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/reprocess/valid/{2 * data_num}.csv',
        f'/mnt/sdb/home/lsr/baseModel_RNA/data/reprocess/valid/{2 * data_num + 1}.csv')

    print("model start...")

    net = ReformerEncoder(2, 2).to(device)
    lr = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion_model = nn.CrossEntropyLoss()

    best_acc = 0
    EPOCH = 100
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


    print('Finished Training...')



# Reformer使用实例
# DE_SEQ_LEN = 4096
# EN_SEQ_LEN = 4096
#
# encoder = ReformerLM(
#     num_tokens = 20000,
#     emb_dim = 128,
#     dim = 1024,
#     depth = 12,
#     heads = 8,
#     max_seq_len = DE_SEQ_LEN,
#     fixed_position_emb = True,
#     return_embeddings = True # return output of last attention layer
# ).cuda()
#
# decoder = ReformerLM(
#     num_tokens = 20000,
#     emb_dim = 128,
#     dim = 1024,
#     depth = 12,
#     heads = 8,
#     max_seq_len = EN_SEQ_LEN,
#     fixed_position_emb = True,
#     causal = True
# ).cuda()
#
# x  = torch.randint(0, 20000, (1, DE_SEQ_LEN)).long().cuda()
# yi = torch.randint(0, 20000, (1, EN_SEQ_LEN)).long().cuda()
#
# enc_keys = encoder(x)               # (1, 4096, 1024)
# yo = decoder(yi, keys = enc_keys)   # (1, 4096, 20000)
