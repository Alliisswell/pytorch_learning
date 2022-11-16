#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/29 18:26
__author__ = '李晓宁'

import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='../dataset_cifar10', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class LiXiaoning(nn.Module):

    def __init__(self):
        super(LiXiaoning, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


lixiaoning = LiXiaoning()
writer = SummaryWriter('../logs006_maxpool')
step = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    writer.add_images('input', imgs, step)
    output = lixiaoning(imgs)
    print(output.shape)
    writer.add_images('maxpool', output, step)
    step += 1

writer.close()

