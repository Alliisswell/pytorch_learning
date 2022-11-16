#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: liuerdaren
# learning time: 2022/10/14 19:04
__author__ = '李晓宁'

import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def target2idx(targets):
    target_idx = []
    target_labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9',
                     'Class_10']
    for target in targets:
        target_idx.append(target_labels.index(target))
    return target_idx


# 1.读取数据
class OttoDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        labels = data['target']
        self.len = data.shape[0]
        self.X_data = torch.tensor(np.array(data)[:, 1:-1].astype(float))
        self.y_data = target2idx(labels)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.len


otto_dataset1 = OttoDataset('download/otto-group-product-classification-challenge/train.csv')
otto_dataset2 = OttoDataset('download/otto-group-product-classification-challenge/test.csv')
train_loader = DataLoader(dataset=otto_dataset1, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=otto_dataset2, batch_size=64, shuffle=False, num_workers=2)


# 2.构建模型
class OttoNet(torch.nn.Module):
    def __init__(self):
        super(OttoNet, self).__init__()
        self.linear1 = torch.nn.Linear(93, 64)
        self.linear2 = torch.nn.Linear(64, 32)
        self.linear3 = torch.nn.Linear(32, 16)
        self.linear4 = torch.nn.Linear(16, 9)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 93)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        x = self.softmax(x)
        return x


ottomodel = OttoNet()
ottomodel







