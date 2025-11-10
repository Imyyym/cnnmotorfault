# -*- coding: utf-8 -*-
# import matplotlib
# matplotlib.use('QT5Agg')
import os
# os.environ[‘HDF5_DISABLE_VERSION_CHECK’] = ‘2’
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
import random

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import scipy.io as scio
import pandas as pd


list_gene=[]#所有基因的列表
gene=[]#单个的基因列表
gene_num=20#基因的初始个数，和每代基因的个数
gene_len=8#每个基因的长度
mutate_prob=0.2#变异概率
group_num = 10  # 小组数
group_size = 10  # 每小组人数
group_winner = int(gene_num / group_num)  # 每小组获胜人数
winners = []  # 锦标赛结果
danwei=(0.5-0.0001)/256
number=0
list_gene_new=[]
zuiqiang=[]
train_shuliang=20
for i in range(gene_num):
    gene =[random.randint(0,1)for j in range(gene_len)]
    random.shuffle(gene)
    list_gene.append(gene)




filepath = '/root/autodl-tmp/Preprocessed_Disconnect_Phase_10_11_21_.mat'
labelname = 'Disconnect_Phase_10_11_21_'
csv_name = '/root/autodl-tmp/Preprocessed_Disconnect_Phase_10_11_21_c.csv'
data_Preprocessed_Disconnect_Phase_10_11_21 = pd.read_csv(csv_name)

filepath = '/root/autodl-tmp/Preprocessed_No_failed.mat'
labelname = 'No_failed'
csv_name = '/root/autodl-tmp/Preprocessed_No_failed_c.csv'
data_Preprocessed_No_failed = pd.read_csv(csv_name)

filepath = '/root/autodl-tmp/Preprocessed_Rotor_Current_Failed_R_.mat'
labelname = 'Rotor_Current_Failed_R_'
csv_name = '/root/autodl-tmp/Preprocessed_Rotor_Current_Failed_R_c.csv'
data_Preprocessed_Rotor_Current_Failed_R = pd.read_csv(csv_name)

filepath = '/root/autodl-tmp/Preprocessed_Short_between_two_phases_.mat'
labelname = 'Short_between_two_phases_'
csv_name = '/root/autodl-tmp/Preprocessed_Short_between_two_phases_c.csv'
data_Preprocessed_Short_between_two_phases = pd.read_csv(csv_name)

filepath = '/root/autodl-tmp/Preprocessed_Test_Data_Rotor_Current_Faild.mat'
labelname = 'Test_Data_Rotor_Current_Faild'
csv_name = '/root/autodl-tmp/Preprocessed_Test_Data_Rotor_Current_Faild_c.csv'
data_Preprocessed_Test_Data_Rotor_Current_Faild = pd.read_csv(csv_name)

filepath = '/root/autodl-tmp/Preprocessed_Test_Data_Short_phases_Ln_G_.mat'
labelname = 'Test_Data_Short_phases_Ln_G_'
csv_name = '/root/autodl-tmp/Preprocessed_Test_Data_Short_phases_Ln_G_c.csv'
data_Preprocessed_Test_Data_Short_phases_Ln_G = pd.read_csv(csv_name)


# dataframe data/  array label
# 这是二维的一个数据处理，data.iloc。将二维变成三维的空间，为下一步做准备，一次实验的准备。
def data_processing(data):
    data = data.iloc[:, 0:8]  # 把标签去掉
    row = data.shape[0]
    data = np.array(data)
    data = data.reshape(int(row / 10000), 10000, 8)  # 是最初的去掉标签的
    return data


def downsample(data, sample_time=4):  # 这里有个采样时间
    row = int(data.shape[0])  # 试验次数
    data1 = np.zeros(
        shape=(int(data.shape[0] * sample_time), int(data.shape[1] / sample_time), data.shape[2]))  # 这是一个加厚了实验次数那一维度
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


# 和4相关了，加厚实验和采样时间减少了一万那个维度的数据，相当于产生了多次实验


'''
label 0  / Test_Data_Rotor_Current_Faild
'''

data_Preprocessed_Test_Data_Rotor_Current_Faild = data_processing(data_Preprocessed_Test_Data_Rotor_Current_Faild)
data_Preprocessed_Test_Data_Rotor_Current_Faild = downsample(data_Preprocessed_Test_Data_Rotor_Current_Faild)
row0 = data_Preprocessed_Test_Data_Rotor_Current_Faild.shape[0]  # 试验次数还是加厚过的
label_data_Preprocessed_Test_Data_Rotor_Current_Faild = 0 * np.ones(shape=(int(row0), 1))  # 打上标签

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
print(data_for_train.shape[0])
print(data_for_train.shape[1])
print(data_for_train.shape[2])

# for test
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

# Nomalizaiton for x_train
row = data_for_train.shape[0]
for i in range(row):
    data_for_train[i:i + 1, :, :] = \
        (data_for_train[i:i + 1, :, :] - np.min(data_for_train[i:i + 1, :, :], axis=1)) / \
        (np.max(data_for_train[i:i + 1, :, :], axis=1) - np.min(data_for_train[i:i + 1, :, :], axis=1))
data_for_train = data_for_train.reshape(data_for_train.shape[0], data_for_train.shape[1], data_for_train.shape[2], 1)

num = np.random.permutation(data_for_train.shape[0])  # 打乱index
data_for_train = data_for_train[num, :]  # 打乱数据
label_for_train = label_for_train[num, :]

row = data_for_test.shape[0]
for i in range(row):
    data_for_test[i:i + 1, :, :] = \
        (data_for_test[i:i + 1, :, :] - np.min(data_for_test[i:i + 1, :, :], axis=1)) / \
        (np.max(data_for_test[i:i + 1, :, :], axis=1) - np.min(data_for_test[i:i + 1, :, :], axis=1))
data_for_test = data_for_test.reshape(data_for_test.shape[0], data_for_test.shape[1], data_for_test.shape[2], 1)


###############################################    model   ###############################################

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()

        self.c1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')  # 卷积层
        # self.c11 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        # self.d1 = Dropout(0.1)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(64, activation='relu')
        self.f2 = Dense(0, activation='relu')
        # self.f4 = Dense(32, activation='relu')
        # self.f3 = Dense(32, activation='relu')
        # self.d2 = Dropout(0.2)
        self.f5 = Dense(6, activation='softmax')

    def call(self, inputs):
        x = self.c1(inputs)
        # x = self.c11(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        # x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        # x = self.f3(x)
        # x = self.f4(x)
        # x = self.d2(x)
        y = self.f5(x)
        return y

fitness_zong=[]
dijici=0
fitness_zong_d=[]
for _ in range(train_shuliang):
    list_gene_lr_shi=[0.0009,0.0009,0.0009,0.0009,0.0009,0.0009]
    list_gene_acc = []

    # 交叉
    for i in range(gene_num - 1):
        index1 = random.randint(0, gene_len - 2)
        index2 = random.randint(index1, gene_len - 1)
        gene_middle = []
        gene_1 = list_gene[i]
        gene_2 = list_gene[i + 1]
        for j in range(index2 - index1 + 1):
            #print(gene_1)
            gene_middle.append(gene_1[index1 + j])
            gene_1[index1 + j] = gene_2[index1 + j]
            gene_2[index1 + j] = gene_middle[j]
    # 变异
    for i in range(gene_num - 1):
        if random.random() < mutate_prob:
            index1_mutate = random.randint(0, gene_len - 2)
            index2_mutate = random.randint(index1_mutate, gene_len - 1)
            gene_mutate = list_gene[i]
            for jj in range(index2_mutate - index1_mutate ):
                if gene_mutate[index1_mutate + jj] == 1:
                    gene_mutate[index1_mutate + jj] = 0
                else:
                    gene_mutate[index1_mutate + jj] = 1
            list_gene[i]=gene_mutate
    # 解码
    list_gene_lr = []
    lr_gene = []
    for i in range(gene_num):
        lr_gene = list_gene[i]
        aa = ''
        for j in range(gene_len):
            aa = aa + str(lr_gene[j])
        bb = int(aa, 2)
        lr = 0.0001 + bb * danwei
        lr_5 = float(format(lr, '.5f'))
        list_gene_lr.append(lr_5)
    for i in list_gene_lr_shi:
        print('lr')
        print(i)
        dijici = dijici + 1
        print('dijici')
        print(dijici)
        model = Baseline()
        adam = Adam(lr=i)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])

        history = model.fit(data_for_train, label_for_train, batch_size=64, epochs=3, validation_split=0.1)
        #model.summary()

        y_pre = model.predict(data_for_test)
        y_pre = np.argmax(y_pre, axis=1).reshape(-1, 1)
        error = label_for_test - y_pre
        num = np.count_nonzero(error)
        accuracy = (1 - num / label_for_test.shape[0]) * 100
        #print('Predictin Accuracy%', accuracy)

        from sklearn.metrics import confusion_matrix

        matrix = confusion_matrix(label_for_test, y_pre)
        #print(matrix)
        ###############################################    show   ###############################################

        # 显示训练集和验证集的acc和loss曲线
        acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        list_gene_acc.append(acc[len(acc) - 1])
        #print(acc[len(acc) - 1])
        #print(list_gene_acc)
        #选择
    group_fitness_mid_mid = []
    for nn in range(group_num):
        group_gene=[]
        group_fitness = []
        group_rank=list(range(gene_num))
        for yy in range(group_size):
        # 随机组成小组
            player_list = []
            player = random.choice(group_rank)
            player_list.append(player)
            player_gene=list_gene[player]
            player_fitness=list_gene_acc[player]
            group_gene.append(player_gene)
            group_fitness.append(player_fitness)
        for k in range(1, group_size-1):
            for l in range(0, group_size - k):
                if group_fitness[l] < group_fitness[l + 1]:
                    group_gene[l], group_gene[l + 1] = group_gene[l + 1], group_gene[l]
                    group_fitness[l], group_fitness[l + 1] = group_fitness[l + 1], group_fitness[l]
        fitness_zong_d.append(group_fitness[0])
        group_fitness_mid=0
        winners = group_gene[0:group_winner]
        list_gene_new = list_gene_new + winners
    min_fitness_zong_d = fitness_zong_d[0]
    fitness_zong.append(min_fitness_zong_d)
    list_gene=list_gene_new
    list_gene_new=[]
    fitness_zong_d=[]

tu = np.linspace(0, train_shuliang, train_shuliang)
plt.plot(tu,fitness_zong)
plt.show()
print(fitness_zong)



