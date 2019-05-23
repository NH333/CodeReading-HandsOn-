#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.rand(100,1)

X_b = np.c_[np.ones((100,1)),X] #加上一列系数为1，是应为x0默认都是1,P148页公式4-1
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new] 
y_predict = X_new_b.dot(theta_best)

plt.plot(X_new,y_predict,"r-")
plt.plot(X,y,"b.")
plt.axis([0,2,0,15])
plt.show()


"""
调用sklearn库的函数，与之前公式计算出来的比较
"""
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print(lin_reg.intercept_) #截距
print(lin_reg.coef_)      #斜率
print(lin_reg.predict(X_new))

"""
批量梯度下降实现
"""
eta = 0.1 #学习率
n_iterations = 1000
m = 100  #m表示迭代次数

theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - eta * gradients     #公式4-7 梯度下降步长

print(theta)

"""
随机梯度下降的简单实现
随机选择一个特征和输出做梯度下降
"""
n_epochs = 50
t0,t1 = 5,50

def learning_schedule(t):
    return t0 / (t + t1)      #t是越来越大的，学习率是越来越低的

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m): #m是之前定义的
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi) #公式4-6 成本函数MSE的梯度向量
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients     #公式4-7 梯度下降步长

from sklearn.linear_model import SGDRegressor #stochastic gradient descend随机梯度下降
sgd_reg = SGDRegressor(n_iter=50,penalty=None,eta0=0.1) #上面随机梯度下降，起始的学习率也为0.1
sgd_reg.fit(X,y.ravel()) #把多维数组降到一维，直接在原数组上操作，而对应的flatten()函数，功能一样，但返回的是一份拷贝，不影响原数组

input()