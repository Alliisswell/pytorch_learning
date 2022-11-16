#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/9/2 13:46
__author__ = '李晓宁'

import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
print(vgg16)
# method1 模型结构+模型参数
torch.save(vgg16, 'vgg16.pth')

# method2 模型参数
torch.save(vgg16.state_dict(), 'vgg16.state_dict.pth')
