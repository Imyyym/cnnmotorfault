# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
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

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scipy.io as scio
import pandas as pd


class Ga:
  def __init__(self):
    self.list_gene = []
    self.gene_num = 0

  def jiaocha(self):
    for i in range(0,gene_num,2):
      index1 = random.randint(0, gene_len - 2)
      index2 = random.randint(index1, gene_len - 1)
      gene_middle=[]
      gene_1=self.list_gene[i]
      gene_2=self.list_gene[i+1]
      for j in range(index2 - index1 + 1):#交叉个数
        gene_middle.append(gene_1[index1 + j])
        gene_1[index1 + j] = gene_2[index1 + j]
        gene_2[index1 + j] = gene_middle[j]
      self.list_gene[i]=gene_1
      self.list_gene[i+1] = gene_2
    return self.list_gene

  def bianyi(self):




list_gene = []  # 所有基因的列表
gene = []  # 单个的基因列表
gene_num = 60  # 基因的初始个数，和每代基因的个数
gene_len = 8  # 每个基因的长度
mutate_prob_b = 0.25  # 变异概率
group_num = 10  # 小组数
group_size = 10  # 每小组人数
group_winner = int(gene_num / group_num)  # 每小组获胜人数
winners = []  # 锦标赛结果
danwei = (0.5 - 0.0001) / 256
number = 0
list_gene_new = []
zuiqiang = []
train_shuliang = 300
for i in range(gene_num):
  gene = [random.randint(0, 1) for j in range(gene_len)]
  random.shuffle(gene)
  list_gene.append(gene)

fitness_zong = []
dijici = 0
fitness_zong_d = []
for _ in range(train_shuliang):
  print('dijici')
  list_gene_acc = []
  dijici = dijici + 1
  # 交叉
  gaa=Ga()
  gaa.list_gene=list_gene
  gaa.gene_num=gene_num
  list_gene=gaa.jiaocha()


  for i in range(gene_num - 1):
    index1 = random.randint(0, gene_len - 2)
    index2 = random.randint(index1, gene_len - 1)
    gene_middle = []
    gene_1 = list_gene[i]
    gene_2 = list_gene[i + 1]
    for j in range(index2 - index1 + 1):
      # print(gene_1)
      gene_middle.append(gene_1[index1 + j])
      gene_1[index1 + j] = gene_2[index1 + j]
      gene_2[index1 + j] = gene_middle[j]
  # 变异
  for i in range(gene_num - 1):
    if random.random() < mutate_prob_b:
      index1_mutate = random.randint(0, gene_len - 2)
      index2_mutate = random.randint(index1_mutate, gene_len - 1)
      gene_mutate_mid = []
      gene_mutate = []
      gene_mutate_mid = list_gene[i]
      for gene_mu in range(gene_len):
        gene_mutate.append(gene_mutate_mid[gene_mu])
      for jj in range(
              index2_mutate - index1_mutate):  # aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        if gene_mutate[index1_mutate + jj] == 1:
          gene_mutate[index1_mutate + jj] = 0
        else:
          gene_mutate[index1_mutate + jj] = 1
      list_gene[i] = gene_mutate
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
  for acc in range(gene_num):
    list_gene_acc.append(list_gene_lr[acc] * 10)
  # 选择
  group_fitness_mid_mid = []
  for nn in range(group_num):
    group_gene = []
    group_fitness = []
    group_rank = list(range(gene_num))
    for yy in range(group_size):
      # 随机组成小组
      player_list = []
      player = random.choice(group_rank)
      player_list.append(player)
      player_gene = list_gene[player]
      player_fitness = list_gene_acc[player]
      group_gene.append(player_gene)
      group_fitness.append(player_fitness)
    for k in range(1, group_size - 1):
      for l in range(0, group_size - k):
        if group_fitness[l] > group_fitness[l + 1]:
          group_gene[l], group_gene[l + 1] = group_gene[l + 1], group_gene[l]
          group_fitness[l], group_fitness[l + 1] = group_fitness[l + 1], group_fitness[l]
    fitness_zong_d.append(group_fitness[0])
    group_fitness_mid = 0
    winners = group_gene[0:group_winner]
    list_gene_new = list_gene_new + winners
  min_fitness_zong_d = fitness_zong_d[0]
  # for min in fitness_zong_d:
  #     if min < min_fitness_zong_d:
  #         min_fitness_zong_d = min
  fitness_zong.append(min_fitness_zong_d)
  # print('asdfasfasdf')
  # print(fitness_zong)
  list_gene = list_gene_new
  list_gene_new = []
  fitness_zong_d = []

tu = np.linspace(0, train_shuliang, train_shuliang)
plt.plot(tu, fitness_zong)
plt.show()

