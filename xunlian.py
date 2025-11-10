# longlonglonglong
import os
# os.environ[‘HDF5_DISABLE_VERSION_CHECK’] = ‘2’
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scipy.io as scio
import pandas as pd


# csv_name_first='D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/1.csv'
# data_train_zong_1_1=pd.read_csv(csv_name_first)
# #print(data_train_zong_1_1.shape)
#
# for i in range(2,51):
#   csv_name = 'D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/'
#   csv_name=csv_name+str(i)+'.csv'
#   data_train=pd.read_csv(csv_name)
#   data_train_value=data_train.values
#   data_train_zong_1_1=np.append(data_train_zong_1_1,data_train_value,axis=0)
# #np.savetxt('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/new1_1.csv', data_train_zong_1_1, delimiter = ',')
#
# csv_name_first='D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/1.csv'
# data_train_zong_1_4=pd.read_csv(csv_name_first)
# for i in range(2,51):
#   csv_name = 'D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/'
#   csv_name=csv_name+str(i)+'.csv'
#   data_train=pd.read_csv(csv_name)
#   data_train_value=data_train.values
#   data_train_zong_1_4=np.append(data_train_zong_1_4,data_train_value,axis=0)
# #np.savetxt('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/new1_4.csv', data_train_zong_1_4, delimiter = ',')


ab=np.ones((1,2))
data_train_zong_1_1=pd.read_csv('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/new1_1.csv')
data_train_zong_1_1=data_train_zong_1_1.values
data_train_zong_1_1=np.append(data_train_zong_1_1,ab,axis=0)
print(data_train_zong_1_1.shape)
data_train_zong_1_1=data_train_zong_1_1.reshape(50,32768,2)
data_train_zong_1_4=pd.read_csv('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/new1_4.csv')
data_train_zong_1_4=data_train_zong_1_4.values
data_train_zong_1_4=np.append(data_train_zong_1_4,ab,axis=0)
data_train_zong_1_4=data_train_zong_1_4.reshape(50,32768,2)
data_train_all=np.append(data_train_zong_1_1,data_train_zong_1_4,axis=0)
data_train_all=data_train_all.reshape(100,32768,2,1)
label1_1=np.ones((50,1))
label1_2=np.ones((50,1))-1
data_train_label_all=np.append(label1_1,label1_2,axis=0)
#print(data_train_label_all)

num = np.random.permutation(data_train_all.shape[0])
data_train_all = data_train_all[num,:] #打乱数据
data_train_label_all = data_train_label_all[num,:]
#print(data_train_label_all)


# csv_name_first='D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/10.csv'
# data_test_zong_1_1=pd.read_csv(csv_name_first)
# #print(data_train_zong_1_1.shape)
#
# for i in range(11,20):
#   csv_name = 'D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/'
#   csv_name=csv_name+str(i)+'.csv'
#   data_test=pd.read_csv(csv_name)
#   data_test_value=data_test.values
#   data_test_zong_1_1=np.append(data_test_zong_1_1,data_test_value,axis=0)
# np.savetxt('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/new1_1_test.csv', data_test_zong_1_1, delimiter = ',')
#
# csv_name_first='D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/10.csv'
# data_test_zong_1_4=pd.read_csv(csv_name_first)
# for i in range(11,20):
#   csv_name = 'D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/'
#   csv_name=csv_name+str(i)+'.csv'
#   data_test=pd.read_csv(csv_name)
#   data_test_value=data_test.values
#   data_test_zong_1_4=np.append(data_test_zong_1_4,data_test_value,axis=0)
# np.savetxt('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/Data/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/new1_4_test.csv', data_test_zong_1_4, delimiter = ',')


data_test_zong_1_1=pd.read_csv('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_1/new1_1_test.csv')
data_test_zong_1_1=data_test_zong_1_1.values
data_test_zong_1_1=np.append(data_test_zong_1_1,ab,axis=0)
data_test_zong_1_1=data_test_zong_1_1.reshape(10,32768,2)
data_test_zong_1_4=pd.read_csv('D:/2024/guzhangjianceshuju/guzhangshuju_nankai/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/XJTU-SY_Bearing_Datasets/35Hz12kN/Bearing1_4/new1_4_test.csv')
data_test_zong_1_4=data_test_zong_1_4.values
data_test_zong_1_4=np.append(data_test_zong_1_4,ab,axis=0)
data_test_zong_1_4=data_test_zong_1_4.reshape(10,32768,2)
data_test_all=np.append(data_test_zong_1_1,data_test_zong_1_4,axis=0)

data_test_all=data_test_all.reshape(20,32768,2,1)
label_test_1_1=np.ones((10,1))
label_test_1_2=np.ones((10,1))-1
data_test_label_all=np.append(label_test_1_1,label_test_1_2,axis=0)
#print(data_train_label_all)

num = np.random.permutation(data_test_all.shape[0])
data_test_all = data_test_all[num,:] #打乱数据
data_test_label_all = data_test_label_all[num,:]
print(data_test_label_all)

####################################################################
####################################################################
####################################################################
####################################################################

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
        self.f2 = Dense(32, activation='relu')
        self.f4 = Dense(32, activation='relu')
        self.f3 = Dense(32, activation='relu')
        # self.d2 = Dropout(0.2)
        #self.f5 = Dense(2, activation='softmax')

        self.f5 = Dense(2, activation='sigmoid')

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
        x = self.f3(x)
        x = self.f4(x)
        # x = self.d2(x)
        y = self.f5(x)
        return y


model = Baseline()
adam = Adam(lr=0.15)
model.compile(optimizer='adam',
                  loss='binary_crossentropy',#'sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

history = model.fit(data_train_all, data_train_label_all, batch_size=4, epochs=40,validation_split=0.1)
model.summary()

y_pre = model.predict(data_test_all)
y_pre = np.argmax(y_pre,axis=1).reshape(-1,1)
error = data_test_label_all-y_pre
num = np.count_nonzero(error)
accuracy = (1-num/data_test_label_all.shape[0])*100
print('Predictin Accuracy%',accuracy)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(data_test_label_all,y_pre)
print(matrix)
###############################################    show   ###############################################

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
print('acc')
print(acc[len(acc)-1])
print('val_acc')
print(val_acc[len(val_acc)-1])
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()





