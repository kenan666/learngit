import numpy as np
import pandas as pd 
from keras.utils import np_utils

from keras.datasets import mnist

# 进行数据集下载
(X_train_image,y_train_image),(X_test_image,y_test_image) = mnist.load_data()

# 读取  mnist 数据集
print('train data = ',len(x_trian_image))
print('test data = ',len(x_test_image))

print('x_train_image : ',x_train_image.shape)
print('y_train_image : ',y_train_image.shape)

#  定义plot_image 函数，显示数字图像
import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image,camp = 'binary')
    plt.show()

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)  #  设置图像大小
    if num>25:
        num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(images[idx],camp = 'binary')
        title = 'label =' + str (label[idx])
        if len (prediction)>0:# 如果传入了 预测结果
            title += ",predict = " + str(prediction[idx])
        
        ax.set_title (title,fontsize = 0)
        ax_set_xticks([]);ax.set_yticks([])
        idx +=1
    plt.show()

    #  查看训练结果
    print('x_test_image:',x_test_image.shape)
    print ('y_test_label :',y_test_label.shape)

#  显示  test 测试结果
    plot_image_labels_prediction(x_test_image,y_test_label,[],0,10)

#  feature 图像预处理
x_Train = x_train_image.reshape((60000,784).astype('float32'))
x_Test = X_test_image.reshape((10000,784).astype('float32'))

print('x_train:',x_Train.shape)
print('x_test:',x_Test.shape)

#  将数字图像images 的数字标准化
x_Train_normalize = x_Train /255
x_Test_normalize = x_Test / 255

x_Train_normalize[0]

#  查看标签字段
y_train_label [:5]

#  label  标签  字段进行 one_hot转换
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

y_TrainOneHot[:5]