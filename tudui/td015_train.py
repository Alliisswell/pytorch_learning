#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# teacher: tudui
# learning time: 2022/9/2 15:04
__author__ = '李晓宁'

import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from td014_model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset_cifar10", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../dataset_cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
print("len(train_data):{0}".format(len(train_data)))
print("len(test_data):{0}".format(len(test_data)))

# 加载数据集
train_load = DataLoader(train_data, batch_size=64)
test_load = DataLoader(test_data, batch_size=64)

# 创建模型
lixiaoning = LiXiaoning()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 创建优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(lixiaoning.parameters(), lr=learning_rate)

# 设置一些训练可视化的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 10

writer = SummaryWriter("../logs015_train")

for i in range(epoch):
    print("----第{}轮训练开始----".format(i + 1))

    # 训练
    lixiaoning.train()
    for data in train_load:  # 每一个bach训练一次
        imgs, targets = data
        outputs = lixiaoning(imgs)

        loss = loss_fn(outputs, targets)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("第{0}次训练，Loss:{1}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss, total_train_step)

    # 测试
    lixiaoning.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 测试的时候不再需要计算梯度
        for data in test_load:
            imgs, targets = data
            outputs = lixiaoning(imgs)

            loss = loss_fn(outputs, targets)
            total_test_loss += loss

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

        total_test_step += 1
        print("整体测试集上的Loss:{}".format(total_test_loss))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        print("整体测试集上的Accuracy:{}".format(total_accuracy / len(test_data)))
        writer.add_scalar("test_accuracy", total_accuracy / len(test_data), total_test_step)

        torch.save(lixiaoning, "lixiaoning_{}.pth".format(i + 1))
        print("第{}轮模型参数已保存".format(i + 1))

writer.close()
