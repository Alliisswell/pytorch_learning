#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/26 15:38
__author__ = '李晓宁'

import torchvision
from tensorboardX import SummaryWriter

data_trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="../dataset_cifar10", train=True, transform=data_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root="../dataset_cifar10", train=False, transform=data_trans, download=True)

# print(train_set[0])
# print(train_set[0][1])
# print(train_set.classes[6])

writer = SummaryWriter("../logs003_dataset_transforms")
for i in range(10):
    img, target = train_set[i]
    writer.add_image("dataset_transforms", img, i)

writer.close()