#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: liuerdaren
# learning time: 2022/10/14 19:04
__author__ = '李晓宁'

# 导入依赖库
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim


# 数据预处理
# 定义函数将类别标签转为id表示，方便后面计算交叉熵
def labels2id(labels):
    target_id = []  # 给所有target建立一个词典
    target_labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8',
                     'Class_9']  # 自定义9个标签
    for label in labels:  # 遍历labels中的所有label
        target_id.append(target_labels.index(label))  # 添加label对应的索引项到target_labels中
    return target_id


class OttogroupDataset(Dataset):  # 准备数据集
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        labels = data['target']
        self.len = data.shape[0]  # 多少行多少列

        # 处理特征和标签
        self.x_data = torch.tensor(np.array(data)[:, 1:-1].astype(float))  # 1:-1 左闭右开
        self.y_data = labels2id(labels)

    def __getitem__(self, index):  # 魔法方法 支持dataset[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 魔法函数 len() 返回长度
        return self.len


# 载入训练集
train_dataset = OttogroupDataset('dataset/otto-group/train.csv')

# 建立数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(93, 64)  # 93个feature
        self.l2 = torch.nn.Linear(64, 32)
        self.l3 = torch.nn.Linear(32, 16)
        self.l4 = torch.nn.Linear(16, 9)
        self.relu = torch.nn.ReLU()  # 激活函数

    def forward(self, x):  # 正向传播
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        return self.l4(x)  # 最后一层不做激活，不进行非线性变换

    def predict(self, x):  # 预测函数
        with torch.no_grad():  # 梯度清零 不累计梯度
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.relu(self.l3(x))
            x = self.relu(self.l4(x))
            # 这里先取出最大概率的索引，即是所预测的类别。
            _, predicted = torch.max(x, dim=1)
            # 将预测的类别转为one-hot表示，方便保存为预测文件。
            y = pd.get_dummies(predicted)
            return y


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 冲量，冲过鞍点和局部最优


def train(epoch):  # 单次循环 epoch决定循环多少次
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data  # 输入数据
        inputs = inputs.float()
        optimizer.zero_grad()  # 优化器归零

        # 前馈+反馈+更新
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # 累计的损失
        if batch_idx % 300 == 299:  # 每300轮输出一次
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


if __name__ == '__main__':
    for epoch in range(3):
        train(epoch)


# 定义预测保存函数，用于保存预测结果。
def predict_save():
    test_data = pd.read_csv('dataset/otto-group/test.csv')
    test_inputs = torch.tensor(
        np.array(test_data)[:, 1:].astype(float))  # test_data是series，要转为array；[1:]指的是第一列开始到最后，左闭右开，去掉‘id’列
    out = model.predict(test_inputs.float())  # 调用预测函数，并将inputs 改为float格式

    # 自定义新的标签
    labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
              'Class_7', 'Class_8', 'Class_9']

    # 添加列标签
    out.columns = labels

    # 插入id行
    out.insert(0, 'id', test_data['id'])
    output = pd.DataFrame(out)
    output.to_csv('my_predict.csv', index=False)
    return output


predict_save()
