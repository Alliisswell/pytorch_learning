#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/8/25 16:14
__author__ = '李晓宁'

from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs002_transforms")
img = Image.open("E:\\PycharmProjects\\pytorch_learning\\tudui\\image\\profile_photo.jpg")
print(img)

# totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(img_tensor)
print('张量形状：', img_tensor.shape)
print('通道数：', len(img_tensor))
print('通道0：\n', img_tensor[0])
print('通道1：\n', img_tensor[1])
print('通道2：\n', img_tensor[2])
writer.add_image("totensor", img_tensor)

# normalize
# print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
# print(img_norm[0][0][0])
writer.add_image("normalize", img_norm)

# resize
trans_resize = transforms.Resize([512, 512])
img_resize = trans_resize(img)
# print(img_resize)
img_resize = trans_totensor(img_resize)
# print(img_resize)
# print(len(img_resize[0]))
writer.add_image("resize", img_resize)

# compose
trans_resize = transforms.Resize(1920)
trans_compose = transforms.Compose([trans_resize, trans_totensor])
img_compose = trans_compose(img)
# print(img_compose)
# print(len(img_compose[0]))
writer.add_image("compose", img_compose)

# random_crop
trans_random = transforms.RandomCrop(20)
trans_compose = transforms.Compose([trans_random, trans_totensor])
# for i in range(10):
#     img_random = trans_compose(img)
#     writer.add_image("random_crop", img_random, i)

writer.close()
