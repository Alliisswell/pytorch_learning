#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/29 13:53
__author__ = '李晓宁'

import torchvision
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="../dataset_cifar10", train=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)


class LiXiaoning(nn.Module):

    def __init__(self):
        super(LiXiaoning, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


lixiaoning = LiXiaoning()

writer = SummaryWriter("../logs005_conv2d")
setp = 0
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    writer.add_images("imgs", imgs, setp)

    output = lixiaoning(imgs)
    print(output.shape)
    output = output.reshape(-1, 3, 30, 30)
    print(output.shape)
    writer.add_images("output", output, setp)
    setp += 1

writer.close()
