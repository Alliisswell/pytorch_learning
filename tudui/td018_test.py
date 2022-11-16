#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 时间：2022/9/3 0:36
__author__ = 'Li Xiaoning'

import torchvision
from PIL import Image
from td014_model import *


img_path = "../image/dog.jpg"
img = Image.open(img_path)
print(img)

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor()])
img = transforms(img)
print(img.shape)
img = torch.reshape(img, (1, 3, 32, 32))
print(img.shape)

model = torch.load("lixiaoning_50.pth", map_location=torch.device('cpu'))
print(model)
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))



