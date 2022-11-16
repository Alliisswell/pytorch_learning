#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/30 13:20
__author__ = '李晓宁'

import torch
import torchvision
from tensorboardX import SummaryWriter

from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader


input = torch.tensor([[1, -0.5],
                      [-1, 3]])
print(input)
input = torch.reshape(input, (-1, 1, 2, 2))
print(input)

dataset = torchvision.datasets.CIFAR10("../dataset_cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class LiXiaoning(nn.Module):
    def __init__(self):
        super(LiXiaoning, self).__init__()
        self.relu1 = ReLU(inplace=False)  # inplace=False 返回一个新对象，原对象不变，避免数据丢失
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        result = self.sigmoid1(x)
        return result


lixiaoning = LiXiaoning()
# output = lixiaoning(input)
# print(output)
writer = SummaryWriter("../logs007_sigmoid")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input_sigmoid", imgs, step)
    output = lixiaoning(imgs)
    writer.add_images("output_sigmoid", output, step)
    step += 1

writer.close()

