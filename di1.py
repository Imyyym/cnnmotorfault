# longlonglonglong
import numpy as np
import matplotlib.pyplot as plt
import random
list_gene=[]#所有基因的列表
gene=[]#单个的基因列表
gene_num=60#基因的初始个数，和每代基因的个数
gene_len=8#每个基因的长度
mutate_prob=0.8#变异概率
group_num = 10  # 小组数
group_size = 10  # 每小组人数
group_winner = gene_num / group_num  # 每小组获胜人数
winners = []  # 锦标赛结果
#编码和解码
#范围是0.0001到0.1，分成256份，2的8次方
#单位
danwei=(0.5-0.0001)/256
#产生初代随机的基因列表



for i in range(gene_num):
    gene =[random.randint(0,1)for j in range(8)]
    random.shuffle(gene)
    list_gene.append(gene)

#生产，交叉变异，解码，cnn，排序，交叉变异，解码，cnn，排序，

#交叉
for i in range(gene_num-1):
    index1 = random.randint(0, gene_len - 2 )
    index2 = random.randint(index1, gene_len - 1)
    gene_middle = []
    gene_1 = list_gene[i]
    gene_2 = list_gene[i + 1]
    for j in range(index2 - index1 + 1):
        print(gene_1)
        gene_middle.append(gene_1[index1 + j])
        gene_1[index1 + j] = gene_2[index1 + j]
        gene_2[index1 + j] = gene_middle[j]

#变异
for i in range(gene_num-1):
    if random.random() < mutate_prob:
        index1_mutate = random.randint(0, gene_len - 2 )
        index2_mutate = random.randint(index1_mutate, gene_len - 1)
        gene_mutate=list_gene[i]
        for j in range(index2_mutate - index1_mutate + 1):
            if gene_mutate[index1_mutate + j] == 1:
                gene_mutate[index1_mutate + j] = 0
            else:
                gene_mutate[index1_mutate + j] = 1

#解码
list_gene_lr=[]
lr_gene=[]
for i in range(gene_num):
    lr_gene=list_gene[i]
    aa = ''
    for j in range(gene_len):
        aa=aa+str(lr_gene[j])
    print(aa)
    bb = int(aa, 2)
    lr = 0.0001 +  bb * danwei
    lr_5=format(lr,'.5f')
    list_gene_lr.append(lr_5)
#print(aa)
print(list_gene_lr)

#CNN




#
# #选择
# for i in range(group_num):
#     group=[]
#     for j in range(group_size):
#     # 随机组成小组
#         player = random.choice(list_gene)
#         player = Individual(player.genes)
#         group.append(player)
#     group = Ga.rank(group)
#     winners += group[:group_winner]











