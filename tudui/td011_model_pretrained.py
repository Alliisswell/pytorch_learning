#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/9/2 10:39
__author__ = '李晓宁'

import torch
import torchvision
from torch import nn

# dataset = torchvision.datasets.ImageNet(root="../dataset_imagenet", split="train", download=True,
#                                         transform=torchvision.transforms.ToTensor())
# train_data = torchvision.datasets.CIFAR10("../dataset_cifar10", train=True, transform=torchvision.transforms.ToTensor(),
#                                           download=True)

vgg16_true = torchvision.models.vgg16(pretrained=True)
torch.save(vgg16_true.state_dict(), 'vgg16_true.pth')
model_true = torch.load('vgg16_true.pth')
print(model_true)

vgg16_false = torchvision.models.vgg16(pretrained=False)
torch.save(vgg16_false.state_dict(), 'vgg16_false.pth')
model_false = torch.load('vgg16_false.pth')
print(model_false)

print('vgg16_true修改前' + '*' * 50 + '\n', vgg16_true)
vgg16_true.classifier.add_module('add_linear', nn.Linear(in_features=1000, out_features=10))  # 针对分类器添加线性层
print('vgg16_true修改后' + '*' * 50 + '\n', vgg16_true)

print('vgg16_false修改前' + '*' * 50 + '\n', vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 修改分类器的第六层
print('vgg16_false修改后' + '*' * 50 + '\n', vgg16_false)
