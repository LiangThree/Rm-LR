# -*- coding: utf-8 -*
# @Time: 2023-03-25 16:08
# @Author: Three
# @File: dataModule.py
# Software: PyCharm
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import pandas as pd


class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)  # size(0) 返回当前张量维数的第一维


def get_npy_DbyDeep(df):
    label_enc = {v: k for k, v in enumerate('ZARNDCQEGHILKMFPSTWYV')}  # Z : 0
    rna_enc = {v: k for k, v in enumerate('AGCTN')}
    # rna_data = []
    # for seq in df.data.values:
    #     flag = 0
    #     for aa in seq:
    #         if aa in rna_enc:
    #             pass
    #         else:
    #             flag = 1
    #     if flag == 0:
    #         rna_data.append([rna_enc[aa] for aa in seq[491:1511]])
    length = 1001
    kmer_len = 50
    rna_data = [[rna_enc[aa] for aa in seq[length // 2 - kmer_len // 2: length // 2 + kmer_len // 2 + 1 ]]
                # todo: 要不要 *3
                for seq in df.data.values]
    # rna_data = [[rna_enc[aa] for aa in seq]
    #                         # todo: 要不要 *3
    #                         for seq in df.data.values]
    # rna_data = [[rna_enc[aa] for aa in seq[491:1511]]
    #             # todo: 要不要 *3
    #             for seq in df.data.values]

    labels = [1 if label else 0 for label in df.label.values]
    return torch.tensor(rna_data), torch.tensor(labels)


def DataLoad(train_data_0_path, train_data_1_path, test_data_0_path, test_data_1_path, valid_data_0_path, valid_data_1_path):
    print("Start data load...")
    train_data_0, train_data_1 = pd.read_csv(train_data_0_path), pd.read_csv(train_data_1_path)
    test_data_0, test_data_1 = pd.read_csv(test_data_0_path), pd.read_csv(test_data_1_path)
    valid_data_0, valid_data_1 = pd.read_csv(valid_data_0_path), pd.read_csv(valid_data_1_path)

    train_rna_data_0, train_label_0 = get_npy_DbyDeep(train_data_0)
    test_rna_data_0, test_label_0 = get_npy_DbyDeep(test_data_0)
    valid_rna_data_0, valid_label_0 = get_npy_DbyDeep(valid_data_0)

    train_rna_data_1, train_label_1 = get_npy_DbyDeep(train_data_1)
    test_rna_data_1, test_label_1 = get_npy_DbyDeep(test_data_1)
    valid_rna_data_1, valid_label_1 = get_npy_DbyDeep(valid_data_1)

    RNA_seq_train = torch.cat((train_rna_data_0, train_rna_data_1))
    RNA_seq_test = torch.cat((test_rna_data_0, test_rna_data_1))
    RNA_seq_valid = torch.cat((valid_rna_data_0, valid_rna_data_1))
    label_train = torch.cat((train_label_0, train_label_1))
    label_test = torch.cat((test_label_0, test_label_1))
    label_valid = torch.cat((valid_label_0, valid_label_1))

    train_dataset = Data.TensorDataset(RNA_seq_train, label_train)
    test_dataset = Data.TensorDataset(RNA_seq_test, label_test)
    valid_dataset = Data.TensorDataset(RNA_seq_valid, label_valid)

    batch_size = 32
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print("Data loaded...")

    return train_iter, test_iter, valid_iter
    # return train_dataset, test_dataset


class pep_data(torch.utils.data.Dataset):
    def __init__(self, *lists):
        self.RNA_seq = lists[0]
        self.label = lists[1]

    def __getitem__(self, index):
        return self.RNA_seq[index], self.label

    def __len__(self):
        return len(self.label[0])
