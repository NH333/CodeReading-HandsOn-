#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris["data"][:,(2,3)]
y = iris["target"]

#添加偏置θ0的系数为1
X_with_bias = np.c_[np.ones([len(X),1]),X]


#随机数的设置一直没搞明白
np.random.seed(2042)

#代替函数train_test_split，手动设置样本，训练数目的设置
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(test_ratio*total_size)
validation_size = int(validation_ratio*total_size)
train_size = total_size - test_size - validation_size

#随机打乱索引，但不改变原来的值
rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]


#把特征转变成独热编码
"""myself"""
def to_one_hot(y):
    result = []
    length = y.max() + 1
    tmp = np.zeros(length)
    for i in range(len(y)):
        tmp[y[i]] = 1
        result.append(tmp)
        tmp = np.zeros(length)
    
    return np.array(result).reshape(-1,length)  #list 没有reshape


"""github"""
# def to_one_hot(y):
#     n_classes = y.max() + 1
#     m = len(y)
#     Y_one_hot = np.zeros((m, n_classes))
#     Y_one_hot[np.arange(m), y] = 1
#     return Y_one_hot

# 测试独热
# test = to_one_hot(y_train[:10])
# y_train[:10]

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)


#softmax函数
#计算概率
def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps,axis=1,keepdims=True)
    return exps / exp_sums 

n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))
# Theta = np.random.randn(n_inputs,n_outputs)


eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)   #计算概率
    loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1)) #利用交叉熵损失函数, axis=1表示每行求平均值
    error = Y_proba - Y_train_one_hot

    #每500次输出一次
    if iteration % 500 == 0:
        print(iteration, loss)
    gradients = 1/m * X_train.T.dot(error) #交叉熵梯度向量
    Theta = Theta - eta * gradients

logits = X_valid.dot(Theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba,axis=1)

#验证集的准确率
print(np.mean(y_predict == y_valid))
y_valid


#加上正则项l2范式
eta = 0.1
n_iterations = 5001
epsilon = 1e-7
m = len(X_train)
alpha = 0.1
best_loss = 1#????


for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    x_loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon),axis=1))
    l2_ = 1/2 * np.sum(np.square(Theta[1:]))
    x_loss_l2 = x_loss + alpha * l2_
    error = Y_proba - Y_train_one_hot

    # if iteration % 500 == 0:
    #     print(iteration,x_loss_l2)
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]] #正则项的偏导数还没理解
    # gradients = 1/m * X_train.T.dot(error)
    Theta = Theta - eta * gradients 

    logits_valid = X_valid.dot(Theta)
    Y_proba_valid = softmax(logits_valid)
    x_valid_loss = -np.mean(np.sum(Y_valid_one_hot * np.log(Y_proba_valid + epsilon),axis=1))
    l2_valid = 1/2 * np.sum(np.square(Theta[1:]))
    x_valid_loss_l2 = x_valid_loss + alpha *  l2_valid

    if iteration % 500 == 0:
        print(iteration,x_valid_loss_l2)


    #早期停止法
    if x_valid_loss_l2 < best_loss:
        best_loss = x_valid_loss_l2
    else:
        print('last iteration is: %d' %(iteration))
        print('best loss:')
        print(best_loss)
        break



# x0, x1 = np.meshgrid(
#         np.linspace(0, 8, 500).reshape(-1, 1),
#         np.linspace(0, 3.5, 200).reshape(-1, 1),
#     )

# # plt.plot(x0,x1,'.')
# plt.plot(x1,'.')
# plt.show()

input()