#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author : Zhenglong Sun
# Data : 2022-1-7 16:12
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pfdataset as pfd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score
from torch.optim import lr_scheduler
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# Hyper-parameters
input_size = 1  #  signal channels
sequence_length = 800  # samples
hidden_size = 128
num_layers = 2
num_classes = 6
num_epochs = 200
batch_size = 100
learning_rate = 0.0001

# parameters for dataset processing
df_path = r'data_all.npy'

pre_disturbance = 400
post_disturbance = 400  #400 300 200
features = (0)  #0'STATOR Voltage_0',,3'STATOR Current_0' ,6'Rotor_Current',7'Speed',

#composed = transforms.Compose([pfd.ToTensor(), pfd.LpNormalize(dim=0), pfd.StdNormalize(dim=0), pfd.ToImage(size=30)])
composed = transforms.Compose([pfd.ToTensor(), pfd.LpNormalize(dim=0), pfd.StdNormalize(dim=0)])
# create dataset
dataset = pfd.MFataset(df_path,  pre_disturbance, post_disturbance, features, transform=composed)


# split the train and test dataset
train_dataset, test_dataset = random_split(dataset, [round(0.8*dataset.__len__()),round(0.2*dataset.__len__())], generator=torch.Generator().manual_seed(7))


# # get first sample and unpack
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

# Fully connected neural network with one hidden layer
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )


    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) #注意乘以2
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) #注意乘以2
        # -> x needs to be: (batch_size, sequence_length, input_size)
        #h0: (2*num_layers, batch_size, hidden layers)
        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc_layers(out)

        return out


model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)  #相当于不衰减


# for drawing
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    model.train() # 模式设为训练模式
    train_loss = 0
    corrects = 0
    train_num = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.type(torch.LongTensor)  #这个很重要，必须有，否则出错
        labels = labels.to(device)

        # Forward pass
        outputs = model.forward(images.float())  #也可以model(images)
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
    #### 验证集测试开始 ####
    # 设置模式为验证模式
    model.eval()
    corrects, test_num, test_loss = 0, 0, 0

    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.type(torch.LongTensor)  #这个很重要，必须有，否则出错
        labels = labels.to(device)
        # Forward pass
        outputs = model.forward(images.float())  #也可以model(images)
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
    images = images.reshape(-1, sequence_length, input_size).to(device)
    labels = labels.type(torch.LongTensor)  #这个很重要，必须有，否则出错
    labels = labels.to(device)
    # Forward pass
    outputs = model.forward(images.float())  #也可以model(images)
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
figure_data.to_csv(
    r'C:\Users\Warrior\Desktop\python 3.8 projects\motor fault classification\results\LSTM figure data '
    r'with speed 200-200.csv')
