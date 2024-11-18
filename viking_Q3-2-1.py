import os
import random
import shutil

def split(train_dir,test_dir,data_dir):
    files = os.listdir(data_dir)    #获取十个类的文件名组成的列表

    for file in files:      #遍历每个类
        file_dir = os.path.join(data_dir,file)  #表示类的地址
        images = os.listdir(file_dir)   #取出一个类的所有图片
        random.shuffle(images)  #对数据集进行重排
        train_son_dir = os.path.join(train_dir, file)
        os.makedirs(train_son_dir,exist_ok=True)     #创建相应类的训练集
        test_son_dir = os.path.join(test_dir, file)
        os.makedirs(test_son_dir,exist_ok=True)     #创建相应类的测试集
        num = int(len(images) * 0.8)    #判断训练集和测试集的分隔点
        image_train = images[:num]      #取出相应的训练集，测试集
        image_test = images[num:]
        # 复制相应的图片到新建的train，test文件夹
        for image in image_train:
            shutil.copy(os.path.join(file_dir,image),train_son_dir)
        for image in image_test:
            shutil.copy(os.path.join(file_dir,image),test_son_dir)

# 数据集路径
data_dir = 'D:\data-picture\dataset'  # 原始数据集路径
train_dir = r'D:\data-picture\train'  # 训练集保存路径
test_dir = r'D:\data-picture\test'  # 测试集保存路径
split(train_dir,test_dir,data_dir)