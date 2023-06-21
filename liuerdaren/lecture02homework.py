#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: liuerdaren
# learning time: 2022/9/22 17:31
__author__ = '李晓宁'

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 这里设函数为y=3x+2
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
w, b = np.meshgrid(W, B)

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    print(y_pred_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

label_font = {
    'color': 'c',
    'size': 15,
    'weight': 'bold'
}

fig = plt.figure()
ax = Axes3D(fig)
fig.add_axes(ax)
ax.plot_surface(w, b, l_sum / 3)

ax.set_xlabel("w axis", fontdict=label_font)
ax.set_ylabel("b axis", fontdict=label_font)
ax.set_zlabel("loss axis", fontdict=label_font)
ax.set_title("loss_w_b", alpha=0.5, color="b", size=50, fontdict=label_font)  # 设置标题，alpha透明度


plt.show()
