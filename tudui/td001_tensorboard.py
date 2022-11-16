#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/25 12:54
__author__ = '李晓宁'

# writer.close()
# 页面刷新后读取的应该是最新的事件文件events.out.tfevents，但是现在这个地方有问题，
# 安装包的时候，把网络代理关掉
# 修改端口 tensorboard --logdir .\logs015_train\ --port 6007


import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

writer = SummaryWriter("../logs001_tensorboard")

image_path = "E:\\PycharmProjects\\pytorch_learning\\tudui\\dataset_ants&bees\\train\\ants_image\\0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("train", img_array, 2, dataformats='HWC')

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 10 * i, i)

writer.close()
