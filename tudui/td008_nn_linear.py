#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/30 13:53
__author__ = '李晓宁'

import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset_cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)


class LiXiaoning(nn.Module):
    def __init__(self):
        super(LiXiaoning, self).__init__()
        self.linear1 = Linear(in_features=196608, out_features=10)

    def forward(self, x):
        result = self.linear1(x)
        return result


lixiaoning = LiXiaoning()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    output = torch.flatten(imgs)
    print(output.shape)  # torch.Size([196608])
    output = lixiaoning(output)
    print(output.shape)  # torch.Size([10])
