# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 11:14:24 2022

@author: Jimmy
"""
# impDEVICES']='0'

# from __future__ import print_function
# import os
import tensorflow as tf
# config = tf.compat.v1.ConfigProto(gpu_options = tf.compat.v1.GPUOptions(allow_growth = True))
# sess = tf.compat.v1.Session(config=config)
# os.environ['CUDA_VISIBLE_DEVICES']='0'

import scipy.io as scio
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('QT5Agg')


def data_load(filepath,labelname,csv_name):
    data = scio.loadmat(filepath)
    for name in data.keys():
        print(name.title())
    data_train = data['train_data']
    data_label = data['label_data']
    data_label_1=np.full((10000,1),data_label[0,0])#标签填充（10000，1）shape
    row=len(data_label)-1
    data_label1 = data_label[0]#表示真正的label
    data_train_pin=np.append(data_train[0,:,:],data_label_1,axis=1)
    # print('len(data_train[0])')
    # print(len(data_train))
    for i in range(row):
        data_train_1=data_train[i+1,:,:]
        data_train_2=np.append(data_train_1,data_label_1,axis=1)
        data_train_pin=np.append(data_train_pin,data_train_2,axis=0)
    data = pd.DataFrame(data_train_pin, \
                        columns=['Voltage_0','Voltage_1','Voltage_2','Current_0',\
                                 'Current_1','Current_2','Rotor_Current','Speed',\
                                     'Failde',labelname])
    data.to_csv(csv_name, index=False)
    #print(type(data))
    return data


filepath = 'faults-D:/2024/guzhangjianceshuju/shuju/motor-main/motor-faults-main/Preprocessed_Disconnect_Phase_10_11_21_.mat'
labelname = 'Disconnect_Phase_10_11_21_'
csv_name = '/root/autodl-tmp/Preprocessed_Disconnect_Phase_10_11_21_c.csv'
data_Preprocessed_Disconnect_Phase_10_11_21 = pd.read_csv(csv_name)

filepath = 'D:/2024/guzhangjianceshuju/shuju/motor-faults-main/motor-faults-main/motor-faults-mainPreprocessed_No_failed.mat'
labelname = 'No_failed'
csv_name = '/root/autodl-tmp/Preprocessed_No_failed_c.csv'
data_Preprocessed_No_failed = pd.read_csv(csv_name)

filepath = 'D:/2024/guzhangjianceshuju/shuju/motor-faults-main/motor-faults-main/Preprocessed_Rotor_Current_Failed_R_.mat'
labelname = 'Rotor_Current_Failed_R_'
csv_name = '/root/autodl-tmp/Preprocessed_Rotor_Current_Failed_R_c.csv'
data_Preprocessed_Rotor_Current_Failed_R = pd.read_csv(csv_name)

filepath = 'D:/2024/guzhangjianceshuju/shuju/motor-faults-main/motor-faults-main/Preprocessed_Short_between_two_phases_.mat'
labelname = 'Short_between_two_phases_'
csv_name = '/root/autodl-tmp/Preprocessed_Short_between_two_phases_c.csv'
data_Preprocessed_Short_between_two_phases = pd.read_csv(csv_name)

filepath = 'D:/2024/guzhangjianceshuju/shuju/motor-faults-main/motor-faults-main/Preprocessed_Test_Data_Rotor_Current_Faild.mat'
labelname = 'Test_Data_Rotor_Current_Faild'
csv_name = '/root/autodl-tmp/Preprocessed_Test_Data_Rotor_Current_Faild_c.csv'
data_Preprocessed_Test_Data_Rotor_Current_Faild = pd.read_csv(csv_name)

filepath = 'D:/2024/guzhangjianceshuju/shuju/motor-faults-main/motor-faults-main/Preprocessed_Test_Data_Short_phases_Ln_G_.mat'
labelname = 'Test_Data_Short_phases_Ln_G_'
csv_name = '/root/autodl-tmp/Preprocessed_Test_Data_Short_phases_Ln_G_c.csv'
data_Preprocessed_Test_Data_Short_phases_Ln_G = pd.read_csv(csv_name)


# feature reshape
def data_processing(data):
    data = data.iloc[:, 0:8]
    row = data.shape[0]
    data = np.array(data)
    data = data.reshape(int(row / 10000), 10000,8)
    return data


def downsample(data, sample_time=4):
    row = int(data.shape[0])
    data1 = np.zeros(shape=(int(data.shape[0] * sample_time), int(data.shape[1] / sample_time), data.shape[2]))
    x1 = np.arange(0, int(data.shape[1]), sample_time)
    x2 = np.arange(1, int(data.shape[1]), sample_time)
    x3 = np.arange(2, int(data.shape[1]), sample_time)
    x4 = np.arange(3, int(data.shape[1]), sample_time)
    for i in range(0, row):
        data1[sample_time * i:sample_time * i + 1, :, :] = data[i:i + 1, x1, :]
        data1[sample_time * i + 1:sample_time * i + 2, :, :] = data[i:i + 1, x2, :]
        data1[sample_time * i + 2:sample_time * i + 3, :, :] = data[i:i + 1, x3, :]
        data1[sample_time * i + 3:sample_time * i + 4, :, :] = data[i:i + 1, x4, :]
    return data1


'''
label 0  / Test_Data_Rotor_Current_Faild
'''
data_Preprocessed_Test_Data_Rotor_Current_Faild = data_processing(data_Preprocessed_Test_Data_Rotor_Current_Faild)
data_Preprocessed_Test_Data_Rotor_Current_Faild = downsample(data_Preprocessed_Test_Data_Rotor_Current_Faild)
row0 = data_Preprocessed_Test_Data_Rotor_Current_Faild.shape[0]
label_data_Preprocessed_Test_Data_Rotor_Current_Faild = 0 * np.ones(shape=(int(row0), 1))

'''
label 1  / Disconnect_Phase_10_11_21
'''

data_Preprocessed_Disconnect_Phase_10_11_21 = data_processing(data_Preprocessed_Disconnect_Phase_10_11_21)
data_Preprocessed_Disconnect_Phase_10_11_21 = downsample(data_Preprocessed_Disconnect_Phase_10_11_21)
row1 = data_Preprocessed_Disconnect_Phase_10_11_21.shape[0]
label_data_Preprocessed_Disconnect_Phase_10_11_21 = 1 * np.ones(shape=(int(row1), 1))

'''
label 2 / Rotor_Current_Failed_R
'''
data_Preprocessed_Rotor_Current_Failed_R = data_processing(data_Preprocessed_Rotor_Current_Failed_R)
data_Preprocessed_Rotor_Current_Failed_R = downsample(data_Preprocessed_Rotor_Current_Failed_R)
row2 = data_Preprocessed_Rotor_Current_Failed_R.shape[0]
label_data_Preprocessed_Rotor_Current_Failed_R = 2 * np.ones(shape=(int(row2), 1))

'''
label 3 / Short_between_two_phases
'''
data_Preprocessed_Short_between_two_phases = data_processing(data_Preprocessed_Short_between_two_phases)
data_Preprocessed_Short_between_two_phases = downsample(data_Preprocessed_Short_between_two_phases)
row3 = data_Preprocessed_Short_between_two_phases.shape[0]
label_data_Preprocessed_Short_between_two_phases = 3 * np.ones(shape=(int(row3), 1))

'''
label 4 / Test_Data_Short_phases_Ln_G
'''
data_Preprocessed_Test_Data_Short_phases_Ln_G = data_processing(data_Preprocessed_Test_Data_Short_phases_Ln_G)
data_Preprocessed_Test_Data_Short_phases_Ln_G = downsample(data_Preprocessed_Test_Data_Short_phases_Ln_G)
row4 = data_Preprocessed_Test_Data_Short_phases_Ln_G.shape[0]
label_data_Preprocessed_Test_Data_Short_phases_Ln_G = 4 * np.ones(shape=(int(row4), 1))

'''
label 5 / No_failed
'''
data_Preprocessed_No_failed = data_processing(data_Preprocessed_No_failed)
data_Preprocessed_No_failed = downsample(data_Preprocessed_No_failed)
row5 = data_Preprocessed_No_failed.shape[0]
label_data_Preprocessed_No_failed = 5 * np.ones(shape=(int(row5), 1))

# pick up the minimum row / rebulid data
row_for_all = int(np.min([row0, row1, row2, row3, row4, row5])) - int(
    0.1 * np.min([row0, row1, row2, row3, row4, row5]))
data_Preprocessed_Test_Data_Rotor_Current_Faild_ = data_Preprocessed_Test_Data_Rotor_Current_Faild[0:row_for_all, :, :]
data_Preprocessed_Disconnect_Phase_10_11_21_ = data_Preprocessed_Disconnect_Phase_10_11_21[0:row_for_all, :, :]
data_Preprocessed_Rotor_Current_Failed_R_ = data_Preprocessed_Rotor_Current_Failed_R[0:row_for_all, :, :]
data_Preprocessed_Short_between_two_phases_ = data_Preprocessed_Short_between_two_phases[0:row_for_all, :, :]
data_Preprocessed_Test_Data_Short_phases_Ln_G_ = data_Preprocessed_Test_Data_Short_phases_Ln_G[0:row_for_all, :, :]
data_Preprocessed_No_failed_ = data_Preprocessed_No_failed[0:row_for_all, :, :]
label_data_Preprocessed_Test_Data_Rotor_Current_Faild_ = label_data_Preprocessed_Test_Data_Rotor_Current_Faild[
                                                         0:row_for_all, :]
label_data_Preprocessed_Disconnect_Phase_10_11_21_ = label_data_Preprocessed_Disconnect_Phase_10_11_21[0:row_for_all, :]
label_data_Preprocessed_Rotor_Current_Failed_R_ = label_data_Preprocessed_Rotor_Current_Failed_R[0:row_for_all, :]
label_data_Preprocessed_Short_between_two_phases_ = label_data_Preprocessed_Short_between_two_phases[0:row_for_all, :]
label_data_Preprocessed_Test_Data_Short_phases_Ln_G_ = label_data_Preprocessed_Test_Data_Short_phases_Ln_G[
                                                       0:row_for_all, :]
label_data_Preprocessed_No_failed_ = label_data_Preprocessed_No_failed[0:row_for_all, :]

data_for_train = np.concatenate(
    [data_Preprocessed_Test_Data_Rotor_Current_Faild_, data_Preprocessed_Disconnect_Phase_10_11_21_, \
     data_Preprocessed_Rotor_Current_Failed_R_, data_Preprocessed_Short_between_two_phases_, \
     data_Preprocessed_Test_Data_Short_phases_Ln_G_, data_Preprocessed_No_failed_], axis=0)
label_for_train = np.concatenate(
    [label_data_Preprocessed_Test_Data_Rotor_Current_Faild_, label_data_Preprocessed_Disconnect_Phase_10_11_21_, \
     label_data_Preprocessed_Rotor_Current_Failed_R_, label_data_Preprocessed_Short_between_two_phases_, \
     label_data_Preprocessed_Test_Data_Short_phases_Ln_G_, label_data_Preprocessed_No_failed_], axis=0)

# for validation
data_Preprocessed_Test_Data_Rotor_Current_Faild_ = data_Preprocessed_Test_Data_Rotor_Current_Faild[row_for_all:row0 + 1,
                                                   :, :]
data_Preprocessed_Disconnect_Phase_10_11_21_ = data_Preprocessed_Disconnect_Phase_10_11_21[row_for_all:row1 + 1, :, :]
data_Preprocessed_Rotor_Current_Failed_R_ = data_Preprocessed_Rotor_Current_Failed_R[row_for_all:row2 + 1, :, :]
data_Preprocessed_Short_between_two_phases_ = data_Preprocessed_Short_between_two_phases[row_for_all:row3 + 1, :, :]
data_Preprocessed_Test_Data_Short_phases_Ln_G_ = data_Preprocessed_Test_Data_Short_phases_Ln_G[row_for_all:row4 + 1, :,
                                                 :]
data_Preprocessed_No_failed_ = data_Preprocessed_No_failed[row_for_all:row5 + 1, :, :]
label_data_Preprocessed_Test_Data_Rotor_Current_Faild_ = label_data_Preprocessed_Test_Data_Rotor_Current_Faild[
                                                         row_for_all:row0 + 1, :]
label_data_Preprocessed_Disconnect_Phase_10_11_21_ = label_data_Preprocessed_Disconnect_Phase_10_11_21[
                                                     row_for_all:row1 + 1, :]
label_data_Preprocessed_Rotor_Current_Failed_R_ = label_data_Preprocessed_Rotor_Current_Failed_R[row_for_all:row2 + 1,
                                                  :]
label_data_Preprocessed_Short_between_two_phases_ = label_data_Preprocessed_Short_between_two_phases[
                                                    row_for_all:row3 + 1, :]
label_data_Preprocessed_Test_Data_Short_phases_Ln_G_ = label_data_Preprocessed_Test_Data_Short_phases_Ln_G[
                                                       row_for_all:row4 + 1, :]
label_data_Preprocessed_No_failed_ = label_data_Preprocessed_No_failed[row_for_all:row5 + 1, :]

data_for_test = np.concatenate(
    [data_Preprocessed_Test_Data_Rotor_Current_Faild_, data_Preprocessed_Disconnect_Phase_10_11_21_, \
     data_Preprocessed_Rotor_Current_Failed_R_, data_Preprocessed_Short_between_two_phases_, \
     data_Preprocessed_Test_Data_Short_phases_Ln_G_, data_Preprocessed_No_failed_], axis=0)
label_for_test = np.concatenate(
    [label_data_Preprocessed_Test_Data_Rotor_Current_Faild_, label_data_Preprocessed_Disconnect_Phase_10_11_21_, \
     label_data_Preprocessed_Rotor_Current_Failed_R_, label_data_Preprocessed_Short_between_two_phases_, \
     label_data_Preprocessed_Test_Data_Short_phases_Ln_G_, label_data_Preprocessed_No_failed_], axis=0)

# Normalizaiton for x_train
row = data_for_train.shape[0]
for i in range(row):
    data_for_train[i:i + 1, :, :] = \
        (data_for_train[i:i + 1, :, :] - np.min(data_for_train[i:i + 1, :, :], axis=1)) / \
        (np.max(data_for_train[i:i + 1, :, :], axis=1) - np.min(data_for_train[i:i + 1, :, :], axis=1))

num = np.random.permutation(data_for_train.shape[0])  # 打乱index
data_for_train = data_for_train[num, :]  # 打乱数据
label_for_train = label_for_train[num, :]

row = data_for_test.shape[0]
for i in range(row):
    data_for_test[i:i + 1, :, :] = \
        (data_for_test[i:i + 1, :, :] - np.min(data_for_test[i:i + 1, :, :], axis=1)) / \
        (np.max(data_for_test[i:i + 1, :, :], axis=1) - np.min(data_for_test[i:i + 1, :, :], axis=1))


class Baseline(tf.keras.Model):
    def __init__(self):
        super(Baseline, self).__init__()

        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(32, return_sequences=False)
        # self.lstm4 = tf.keras.layers.LSTM(64, return_sequences = False)

        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(256, activation='relu')
        self.dense4 = tf.keras.layers.Dense(256, activation='relu')
        # self.dense5 = tf.keras.layers.Dense(256, activation='relu')
        self.dense6 = tf.keras.layers.Dense(6, activation='softmax')

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        # x = self.lstm4(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        # x = self.dense5(x)
        outputs = self.dense6(x)
        return outputs


model = Baseline()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

history = model.fit(data_for_train, label_for_train, batch_size=16, epochs=4, validation_split=0.1)
model.summary()

y_pre = model.predict(data_for_test)
y_pre = np.argmax(y_pre, axis=1).reshape(-1, 1)
error = label_for_test - y_pre
num = np.count_nonzero(error)
accuracy = (1 - num / label_for_test.shape[0]) * 100
print('Predictin Accuracy%', accuracy)

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(label_for_test, y_pre)
print(matrix)

from matplotlib import pyplot as plt
# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

