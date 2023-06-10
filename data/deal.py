# -*- coding: utf-8 -*
# @Time: 2023-03-30 20:00
# @Author: Three
# @File: deal.py
# Software: PyCharm

import pandas as pd
from tqdm import trange

dataset_use = "test"
# dataset_use = "train"
# dataset_use = "valid"
print(f"dataset_use:{dataset_use}")

# 打开文件

length = 128
coincide = length // 2

for i in range(20):
    with open(f'/mnt/sdb/home/lsr/baseModel_RNA/data/reprocess/{dataset_use}/{i}.csv', 'r', encoding="utf-8-sig") as f_in:
        print("current:", i)
        data = pd.read_csv(f_in, encoding="utf-8-sig")
        length = 128 * 2
        all_length = len(data['data'][0]) * 2
        # print(all_length)
        # data_cut_column = all_length // length + 1
        data_cut_column = all_length // length + 1
        current_column = [[] for i in range(data_cut_column)]

        data_loc = 2
        # 读取文件
        for j in trange(len(data)):
            # data['data'][j] = data['data'][j][all_length//2-510//2 : all_length//2+510//2]
            data.loc[j, 'data'] = data.loc[j, 'data'].replace(" ", "")

        if i % 2 == 0:
            data['label'] = 0
        elif i % 2 == 1:
            data['label'] = 1

        # data['data'] = ' '.join(data['data'])
        # print(data['data'][0])
        # data.drop(labels='idx', axis=1)

        data.columns = ['idx', 'data', 'label']
        data.to_csv(f'/mnt/sdb/home/lsr/baseModel_RNA/data/process/{dataset_use}/{i}.csv', index=False)
