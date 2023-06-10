import dnachisel
import pandas as pd
from dnachisel.biotools import reverse_translate
import random

"""将数据集转换 逆转录为RNA序列"""
def reverseTranslate(peptide):
    # 逆转录为RNA序列
    rna = reverse_translate(peptide, randomize_codons=True, table="Standard")
    return rna

def changeDataset(path):
    # 读取数据集
    dataset = pd.read_csv(path)
    # 获取peptide序列
    peptide = dataset["peptide"]
    # 逆转录为RNA序列
    rna = [reverseTranslate(peptide[i]) for i in range(len(peptide))]
    # 将RNA序列添加到数据集中
    dataset["RNA"] = rna
    # 去除 nterm cterm miss1 miss2 序列
    dataset.drop("nterm", axis=1, inplace=True)
    dataset.drop("cterm", axis=1, inplace=True)
    dataset.drop("miss1", axis=1, inplace=True)
    dataset.drop("miss2", axis=1, inplace=True)
    # dataset.drop("peptide", axis=1, inplace=True)
    # 随机生成标签
    labels = dataset["label"]
    changeLabel = lambda x : 1 if x == True else 0
    dataset["label"] = [changeLabel(labels[i]) for i in range(len(labels))]
    # 保存数据集
    # print(dataset)
    dataset.to_csv(f"./dataset/new_{path.split('.csv')[0].split('/')[-1]}.csv", index=False)

if __name__ == '__main__':
    """测试逆转录"""
    # # 定义peptide序列
    # peptide = "LLE"
    # # 逆转录为RNA序列
    # rna = reverse_translate(peptide, randomize_codons=True, table="Standard")
    # # 打印结果
    # print(rna)
    """测试数据集转换"""
    changeDataset('./dataset/train.csv')
    changeDataset('./dataset/test.csv')
    print("转换完成...")
