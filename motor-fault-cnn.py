#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author : Zhenglong Sun
# Data : 2022-1-7 16:12
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import pandas as pd

import pfdataset as pfd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters

num_classes = 6
num_epochs =200
batch_size = 100
learning_rate = 0.0001


# parameters for dataset processing

df_path = r'data_all.npy'

pre_disturbance = 300
post_disturbance = 300
features = (6,7)
#composed = transforms.Compose([pfd.ToTensor(), pfd.LpNormalize(dim=0), pfd.StdNormalize(dim=0), pfd.ToImage(size=30)])
composed = transforms.Compose([pfd.ToTensor(), pfd.LpNormalize(dim=0), pfd.ToImage(size=30)])
# create dataset
dataset = pfd.MFataset(df_path,  pre_disturbance, post_disturbance, features, transform=composed)

# split the train and test dataset
train_dataset, test_dataset = random_split(dataset, [round(0.8 * dataset.__len__()), round(0.2 * dataset.__len__())],
                                           generator=torch.Generator().manual_seed(7))

# get first sample and unpack
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

# Data loader
# has shuffled in df
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# examples = iter(test_loader)
# example_data, example_targets = examples.next()

# for i in range(6):
#     print(example_targets[i])
#     plt.subplot(2,3,i+1)
#     plt.plot(example_data[i][0:100], 'ro')
# plt.show()

# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2,  #这个channel得根据特征数改变下
                            out_channels=16,
                            kernel_size=2,
                            stride=1,
                            padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 2, 1, 0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 2, 1, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 1, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2)
        )
        self.mlp1 = torch.nn.Linear(26 * 16 * 64, 64)
        # 输出矩阵大小为x、输入矩阵大小为n、卷积核大小为f、步长为s、padding 填充为p x=（n-f+2p）/s +1
        self.mlp2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


model = CNNnet().to(device)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


model.apply(weights_init)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

# for drawing
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    model.train()  # 模式设为训练模式
    train_loss = 0
    corrects = 0
    train_num = 0
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.to(device)
        labels = labels.type(torch.LongTensor)  # 这个很重要，必须有，否则出错
        labels = labels.to(device)

        # Forward pass
        outputs = model.forward(images.float())  # 也可以model(images)
        pre_lab = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        corrects += torch.sum(pre_lab == labels.data)
        train_num += images.size(0)

        # LR Decays
        scheduler.step()

    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(corrects.double().item() / train_num)
    print("Epoch{}, Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
    # 设置模式为验证模式
    model.eval()
    corrects, test_num, test_loss = 0, 0, 0

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.type(torch.LongTensor)  # 这个很重要，必须有，否则出错
        labels = labels.to(device)
        # Forward pass
        outputs = model.forward(images.float())  # 也可以model(images)
        pre_lab = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        corrects += torch.sum(pre_lab == labels)
        test_num += images.size(0)

    # 计算经过一个epoch的训练后再测试集上的损失和精度
    test_loss_all.append(test_loss / test_num)
    test_acc_all.append(corrects.double().item() / test_num)

    print("Epoch{} Test Loss: {:.4f} Test Acc: {:.4f}".format(epoch, test_loss_all[-1], test_acc_all[-1]))

plt.figure(figsize=[14, 5])
plt.subplot(1, 2, 1)
plt.plot(train_loss_all, "ro-", label="Train Loss")
plt.plot(test_loss_all, "bs-", label="Val Loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc_all, "ro-", label="Train Acc")
plt.plot(test_acc_all, "bs-", label="Test Acc")
plt.xlabel("epoch")
plt.ylabel("Acc")
plt.legend()

plt.show()

# 最后输出模型的精度

predict_labels = []
true_labels = []

for step, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.type(torch.LongTensor)  # 这个很重要，必须有，否则出错
    labels = labels.to(device)
    # Forward pass
    outputs = model.forward(images.float())  # 也可以model(images)
    pre_lab = torch.argmax(outputs, 1)
    predict_labels += pre_lab.flatten().tolist()
    true_labels += labels.flatten().tolist()

print(classification_report(predict_labels, true_labels, digits=4))
print("Accuracy of the network：", accuracy_score(predict_labels, true_labels))
figure_data = pd.concat(
    [pd.DataFrame({'train_loss_all': train_loss_all}), pd.DataFrame({'test_loss_all': test_loss_all}),
     pd.DataFrame({'train_acc_all': train_acc_all}), pd.DataFrame({'test_acc_all': test_acc_all}),
     pd.DataFrame({'predict_labels': predict_labels}), pd.DataFrame({'true_labels': true_labels})]
    , axis=1)
figure_data.to_csv(r'C:\Users\Warrior\Desktop\python 3.8 projects\motor fault classification\results\Two features CNN '
                   r'figure '
                   r'data '
                   r'with rotor current and speed 300-300.csv')