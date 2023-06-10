# -*- coding: utf-8 -*
# @Time: 2023-04-17 18:58
# @Author: Three
# @File: data_reprocess.py
# Software: PyCharm

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def split_data():
    for num in range(12):
        print(f"current:{num}")
        data1 = pd.read_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/train/{2 * num}.csv')
        data2 = pd.read_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/train/{2 * num + 1}.csv')
        data3 = pd.read_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/valid/{2 * num}.csv')
        data4 = pd.read_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/valid/{2 * num + 1}.csv')
        data5 = pd.read_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/test/{2 * num}.csv')
        data6 = pd.read_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/raw/test/{2 * num + 1}.csv')

        df_0 = pd.concat([data1, data3, data5])
        df_1 = pd.concat([data2, data4, data6])

        count_0 = len(df_0)
        count_1 = len(df_1)

        train_count_0 = int(np.ceil(0.8 * count_0))
        valid_count_0 = int((count_0 - train_count_0) // 2)
        test_count_0 = int(count_0 - train_count_0 - valid_count_0)

        train_count_1 = int(np.ceil(0.8 * count_1))
        valid_count_1 = int((count_1 - train_count_1) // 2)
        test_count_1 = int(count_1 - train_count_1 - valid_count_1)

        data_0 = shuffle(df_0)
        data_1 = shuffle(df_1)

        data_0[:train_count_0].to_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/train/{2 * num}.csv',
                                      index=False, header=True)
        data_0[train_count_0:train_count_0+valid_count_0].to_csv(
            f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/valid/{2 * num}.csv', index=False, header=True)
        data_0[train_count_0+valid_count_0:train_count_0+valid_count_0+test_count_0].to_csv(
            f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/test/{2 * num}.csv', index=False, header=True)

        data_1[:train_count_1].to_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/train/{2 * num + 1}.csv',
                                      index=False, header=True)
        data_1[train_count_1:train_count_1+valid_count_1].to_csv(
            f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/valid/{2 * num + 1}.csv', index=False, header=True)
        data_1[train_count_1+valid_count_1:train_count_1+valid_count_1+test_count_1].to_csv(
            f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/test/{2 * num + 1}.csv', index=False, header=True)


split_data()
