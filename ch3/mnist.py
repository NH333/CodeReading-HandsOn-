#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8
# coding: unicode_escape

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib as mpl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#mpl.use('Agg')




mnist = fetch_mldata('MNIST original')
#print(mnist)

X = mnist["data"]
y = mnist["target"]

some_digit = X[36000]
print(len(some_digit))

some_digit_image = some_digit.reshape(28,28)
plt.matshow(some_digit_image,cmap=plt.cm.gray)
plt.show()
#print(some_digit_image)

#plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation='nearest')
plt.axis("off")
#plt.show()

#---- X:smpale---y:label ----#
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

#permutation 打乱一�?���?
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

######################
###�?��一�?��元分类器###
######################


#数字5检测器
y_train_5 = (y_train == 5) 
print(len(y_train_5))
y_test_5 = (y_test == 5)

#SGD随机�?��下降分类�?
sgd_clf = SGDClassifier(random_state = 42) 
sgd_clf.fit(X_train,y_train_5)

#result = sgd_clf.predict([some_digit])
#print(result)


#n_splits=3 ??????????????3?? ??????test?????????????-??????train???????
skfolds = StratifiedKFold(n_splits=3, random_state=4, shuffle=False)

for train_index, test_index in skfolds.split(X_train,y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index] #folds??????????????index????????
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    print('Train: %s | test: %s' % (train_index,test_index),'\n')
    print('lenTrain: %d | lenText: %d' %(len(train_index),len(test_index)),'\n')

    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_fold) #???????????
    n_correct = sum(y_pred == y_test_fold)  #????????????
    print(n_correct)                        #??????????20000??
    print(n_correct/len(y_pred))


print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)
print(y_train_pred)

result_CFusionMat = confusion_matrix(y_train_5,y_train_pred)
print(result_CFusionMat)

#??????????
result_Precision_score = precision_score(y_train_5,y_train_pred)
result_Recall_socre = recall_score(y_train_5,y_train_pred)
print('TP/(TP+FP) = %.4f | TP(TP+FN) = %.4f '% (result_Precision_score,result_Recall_socre),'\n')

###################
### ?????????? ###
###################

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function") #??????????????
#print(y_scores) 

##########################
###??????????????????###
##########################

"""
precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores) #?????????????????????????? ????? ??????????


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
  
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()
"""


fpr, tpr, thresholds = roc_curve(y_train_5,y_scores)
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,linewidth=2,label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr,tpr)
#plt.show()


forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_5,cv=3,method="predict_proba") #????predict_proba ??????????? ????????????????????????????????
print(y_probas_forest)

y_scores_forest = y_probas_forest[:,1]
fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend(loc="bottom right")
#plt.show()

###################################################################################################################


"""
�??��计算的精度和�??���??
0.5�??��字为5的阈值�?�??
大于0.5就当作数字为5
""" 
y_scores_forest_copy = y_scores_forest.copy()
y_scores_forest_copy[y_scores_forest_copy>=0.5] = 1
y_scores_forest_copy[y_scores_forest_copy<0.5] = 0

result_Precision_score_forest = precision_score(y_train_5,y_scores_forest_copy)
result_Recall_socre_forest = recall_score(y_train_5,y_scores_forest_copy)
print('TP/(TP+FP) = %.4f | TP(TP+FN) = %.4f '% (result_Precision_score_forest,result_Recall_socre_forest),'\n')
###################################################################################################################



###############
###多元分类�??###
###############
sgd_clf.fit(X_train,y_train)
sgd_clf.predict([some_digit])

#决策分数
some_digit_scores = sgd_clf.decision_function([some_digit])
#找到分数最高的标�?�??
np.argmax(some_digit_scores)



#####错�?分析####
################
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred)#混淆矩阵 行：实例，列：�?测的类别

print(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
#plt.show()

row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
plt.show()


#################
####多标签分�??####
#################

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_mutilabel = np.c_[y_train_large,y_train_odd]

knn_clf = KNeighborsClassifier()     #KNeighborsClassifier �??��多标签分�??
knn_clf.fit(X_train,y_mutilabel)

result_knn_muti = knn_clf.predict([some_digit])
print(result_knn_muti)


######################
###多输�??-多类�??���??###
######################
import numpy.random as rnd
noise_Train = rnd.randint(0,100,(len(X_train),784))
noise_Test = rnd.randint(0,100,(len(X_test),784))
X_train_mod = X_train + noise_Train
X_test_mod = X_test + noise_Test
y_train_mod = X_train
y_test_mod = X_test


# plt.matshow(X_train_mod[36000].reshape(28,28),cmap=plt.cm.gray)
# plt.show()
some_index = 10
knn_clf.fit(X_train_mod,y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plt.matshow(clean_digit.reshape(28,28),cmap=plt.cm.gray)
plt.matshow(y_test_mod[some_index].reshape(28,28),cmap=plt.cm.gray)
plt.show()


input()