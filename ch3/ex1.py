#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8


from sklearn.datasets import fetch_mldata
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
mnist = fetch_mldata('MNIST original')
X = mnist["data"]
y = mnist["target"]


# X_train = X[:60000]
# y_train = y[:60000]
# X_test = X[60000:]
# y_test = y[60000:]

shuffle_index = np.random.permutation(70000)
# X_train = X[shuffle_index[:60000]]
# y_train = y[shuffle_index[:60000]]
# X_test = X[shuffle_index[60000:]]
# y_test = y[shuffle_index[60000:]]


X_train = X[shuffle_index[:600]]
y_train = y[shuffle_index[:600]]
X_test = X[shuffle_index[600:700]]
y_test = y[shuffle_index[600:700]]

print(len(X_train))
print(len(y_train))
param_grid = [{'weights':["uniform","distance"],'n_neighbors':[3,4,5]}]
"""不知道为啥，选择上面uniform和distance 就会报错  原来是uniform 拼写错了 unifrom了。。。。"""
#param_grid = [{'weights':["uniform"],'n_neighbors':[3,4,5]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf,param_grid,cv=5,verbose=3,n_jobs=-1) #cv交叉验证参数

grid_search.fit(X_train,y_train)
best_param = grid_search.best_params_
print(best_param)
print(grid_search.best_score_)

from sklearn.metrics import accuracy_score
y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test,y_pred))

input()