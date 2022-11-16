#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/9/2 14:42
__author__ = '李晓宁'

import torch
from torch import nn


# 定义网络模型
class LiXiaoning(nn.Module):
    def __init__(self):
        super(LiXiaoning, self).__init__()
        self.model = nn.Sequential(
            # Feature Extraction，由卷积层，池化层（下采样层）组成
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=2),  # 图像w,h缩小一倍，c不变
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.MaxPool2d(kernel_size=2),
            # Classification，由全连接层（线性层）组成
            nn.Flatten(),  # 将特征提取结果转化为向量
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        result = self.model(x)
        return result


# 检验模型正确性
if __name__ == '__main__':
    lixiaoning = LiXiaoning()
    input = torch.ones((64, 3, 32, 32))
    output = lixiaoning(input)
    print(output.shape)
