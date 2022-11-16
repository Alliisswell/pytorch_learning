#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/9/2 13:47
__author__ = '李晓宁'

import torch
import torchvision

# 加载method1 模型结构+模型参数
model1 = torch.load('vgg16.pth')
print(model1)

# 加载method2 模型参数
model2 = torch.load('vgg16.state_dict.pth')
print(model2)
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(model2)
# print(vgg16)


