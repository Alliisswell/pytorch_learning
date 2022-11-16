#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: liuerdaren
# learning time: 2022/10/10 13:49
__author__ = '李晓宁'

import math

import torch

pred = torch.tensor([[-0.2], [0.2], [0.8]])
target = torch.tensor([[0.0], [0.0], [1.0]])

sigmoid = torch.nn.Sigmoid()
pred_s = sigmoid(pred)
print(pred_s)
"""
pred_s 输出tensor([[0.4502],[0.5498],[0.6900]])
0*math.log(0.4502)+1*math.log(1-0.4502)
0*math.log(0.5498)+1*math.log(1-0.5498)
1*math.log(0.6900) + 0*log(1-0.6900)
"""
result = 0
i = 0
for label in target:
    if label.item() == 0:
        result += math.log(1 - pred_s[i].item())
    else:
        result += math.log(pred_s[i].item())
    i += 1
result /= 3
print("bce：", -result)
loss = torch.nn.BCELoss()
print('BCELoss:', loss(pred_s, target).item())
