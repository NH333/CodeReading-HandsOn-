#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
list(iris.keys())

print(iris.DESCR)
# print(iris["data"])

X = iris["data"][:, (2,3)]  
y = (iris["target"] == 2).astype(np.int)

# print(X)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="liblinear",C=10**10,random_state=42)
log_reg.fit(X,y)

x0,x1 = np.meshgrid(
    np.linspace(2.9,7,500).reshape(-1,1),
    np.linspace(0.8,2.7,200).reshape(-1,1),
)
X_new = np.c_[x0.ravel(),x1.ravel()]
y_proba = log_reg.predict_proba(X_new)


plt.figure(figsize=(10,4))
plt.plot(X[y==0,0],X[y==0,1],"bs")
plt.plot(X[y==1,0],X[y==1,1],"g^")

zz = y_proba[:,1].reshape(x0.shape)

contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
# save_fig("logistic_regression_contour_plot")
plt.show()
# X_new = np.linspace(0,3,1000).reshape(-1,1)
# y_proba = log_reg.predict(X_new)

# y_proba.reshape(-1,2)

# plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
# plt.plot(X_new, y_proba, "b--", linewidth=2, label="Not Iris-Virginica")

plt.show()
input()