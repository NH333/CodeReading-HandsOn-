#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

m = 100
X = 6 * np.random.rand(m,1) - 3  #m*1的数组  范围-3到3
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)

plt.plot(X,y,"b.")  #"b.".表示散点图，b表示蓝色


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2,include_bias=False) #?
X_poly = poly_features.fit_transform(X) #X_poly 包含了原本的特征 和 该特征的平方

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
print(lin_reg.intercept_,lin_reg.coef_)

X_new = np.linspace(-3,3,100).reshape(100,1) #生产-3到3之间的等差数列，100个
X_new_poly = poly_features.fit_transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X_new,y_new,"r-")
plt.show()

from sklearn.metrics import mean_squared_error #均方误差
from sklearn.model_selection import train_test_split

def plot_learning_curves(model,X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
    
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="val")
    plt.legend(loc="upper right", fontsize=14)
    # plt.show()
    plt.xlabel("Training set size", fontsize=14) 
    plt.ylabel("RMSE", fontsize=14) 

lin_reg = LinearRegression()    
plot_learning_curves(lin_reg,X,y)
plt.axis([0,80,0,3]) #[xmin,xmax,ymin,ymax]
plt.show()

"""
10阶多项式模型的学习曲线
"""
from sklearn.pipeline import Pipeline
polynomial_linearRegression = Pipeline((
    ("poly_features",PolynomialFeatures(degree=10,include_bias=False)),
    ("sgd_reg",LinearRegression()),
))

# plot_learning_curves(polynomial_linearRegression,X,y)
# plt.axis([0,80,0,3]) #[xmin,xmax,ymin,ymax]
# plt.show()
input()