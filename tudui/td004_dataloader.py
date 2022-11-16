#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/29 10:20
__author__ = '李晓宁'

import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

test_dataset = torchvision.datasets.CIFAR10(root="../dataset_cifar10", train=False,
                                            transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

img, target = test_dataset[2]
print(img.shape)
print(target)

writer = SummaryWriter("../logs004_dataloader")
for epoch in range(4):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)
        writer.add_images("epoch_shuffle:{}".format(epoch), imgs, step)
        step += 1

writer.close()
