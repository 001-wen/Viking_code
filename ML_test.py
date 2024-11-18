import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
x=10*np.random.rand(500,1)
y=3*x+4+np.random.randn(500,1)

class Linear_model:
    def __init__(self,learning_rate,literation_number=1000):
        self.learning_rate=learning_rate
        self.literation_number=literation_number
        self.w=None
        self.b=None
        self.loss_lists=[]
        self.loss=None
    def fit(self,x,y):
        lenth=len(x)
        self.w=np.random.randn()
        self.b=np.random.randn()
        for i in range(self.literation_number):
            y_pred=self.w*x+self.b
            self.loss=np.sum((y_pred-y)**2)/lenth
            self.loss_lists.append(self.loss)
            self.w-=2*(y-y_pred)*x*self.learning_rate
            self.b-=2*(y-y_pred)*self.learning_rate
    def pic_literation_number_loss(self):
        fig1=plt.figure(figsize=(10,6))
        plt.plot(self.loss_lists,c='skyblue',ls='--',label='literation line')
        plt.title('literation-loss')
        plt.xlabel('literation-number')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

def pic_learning_rate_loss(learing_rate_list,x,y):
    fig2 = plt.figure(figsize=(10, 6))
    for lr in learing_rate_list:
        model_lr=Linear_model(lr)
        model_lr.fit(x, y)
        plt.plot(model_lr.loss_lists,label=f'learing_rate={lr}')
    plt.title('learning_rate-loss')
    plt.xlabel('learing-rate')
    plt.ylabel('loss')
    plt.legend()
    plt.show()



model=Linear_model(0.01)
model.fit(x, y)
model.pic_literation_number_loss()
lr_lists=[0.001,0.005,0.01]
pic_learning_rate_loss(lr_lists,x,y)


#from sklearn.metrics import mean_squared_error
# mse = mean_squared_error(y_test, y_predict)
# print(f'Mean Squared Error: {mse:.2f}')


# import torch
# import torch.nn as nn
#
# def conv(in_channel,kernel):
#     kernel = nn.Parameter(nn.randn(5,5))
#     output = torch.zeros(24,24)
#     for h in range(24):
#         for w in range(24):
#             inputs = in_channel[h:h+5,w:w+5]
#             output[h,w] = (inputs*kernel).sum()
#     return output
# m = nn.Conv2d(1,1,(5,5),bias=False)
# x = torch.randn(1,1,28,28)