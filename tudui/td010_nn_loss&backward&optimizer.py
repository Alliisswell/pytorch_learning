#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/31 10:38
__author__ = '李晓宁'

import torch.optim
import torchvision

from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset_cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class LiXiaoning(nn.Module):
    def __init__(self):
        super(LiXiaoning, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        result = self.model1(x)
        return result


lixiaoning = LiXiaoning()
loss = nn.CrossEntropyLoss()
optimzer = torch.optim.SGD(lixiaoning.parameters(), lr=0.01)
for i in range(20):
    running_loss = 0.
    for data in dataloader:
        imgs, targets = data
        outputs = lixiaoning(imgs)
        result_loss = loss(outputs, targets)  # 计算数据集每一个bach的损失

        optimzer.zero_grad()  # 重置梯度为零
        result_loss.backward()  # 反向传播，求出损失函数的梯度
        optimzer.step()  # 优化器根据所求梯度更新权重
        running_loss += result_loss
    print(running_loss)

