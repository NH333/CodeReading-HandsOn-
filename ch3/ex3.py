#!/usr/bin/python
# -*- coding: utf-8 -*-       
#coding=utf-8

import os
import matplotlib.pyplot as plt
TITANIC_PATH = os.path.join('C:/Users/84241/Desktop/vscode_Python/Hands-On/ch3/','titanic/')#路径结合

import pandas as pd

def load_titanic_data(filename,titanic_path = TITANIC_PATH):
    csv_path = os.path.join(titanic_path,filename)
    return pd.read_csv(csv_path)

train_data = load_titanic_data("train.csv") 
test_data = load_titanic_data("test.csv")


print(train_data.head())#查看前五个数据
print(train_data.info())#查看每类数据的具体总数，类型
print(train_data.describe())#可以看到数值的具体属性，包括均值，最大最小值等

print(train_data["Embarked"].value_counts())#告诉我们乘客在哪里上船:C=瑟堡，Q=昆士敦，S=南安普敦

from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
#DataFrame数据选择器
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

from sklearn.pipeline import Pipeline
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

"""数值特征的管道"""
num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
#对缺失的数据用imputer进行填充，方式是：均值填充

num_pipeline.fit_transform(train_data)
print(num_pipeline.fit_transform(train_data))

"""需要填充的就只有Embarked，而且只缺了两个值"""
"""所以就选择其中最多的种类作为缺失值"""
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

"""分类特征的管道"""
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])
#具体列的排序方法是 Pclass：1，2，3；Sex：female，male；Embarked：C，Q，S；
#采用的是独热编码
data = cat_pipeline.fit_transform(train_data)


"""最后，我们将之前做好的数值特征管道和分类特征管道进行组合，
获得一个综合管道，该综合管道既可以处理数值特征，又可以处理分类特征。
调用FeatureUnion函数来进行特征管道的组合"""
from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
X_train = preprocess_pipeline.fit_transform(train_data)
print(X_train[0])

# print(train_data["Pclass"].value_counts())
# print(data[2]) #8列应该是因为Place 有3个选项，sex有两种，Embarked有3个选项
# print(train_data.loc[2])


"""进入训练模型"""
"""由于是分类生存还是遇难，是二分类问题"""
y_train = train_data["Survived"]

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto") #其中gamma="auto"表示，如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features。总之，这个参数过大会过拟合，过小会欠拟合，一般都取“auto”。
svm_clf.fit(X_train,y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred_svm = svm_clf.predict(X_test)

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf,X_train,y_train,cv=10)
print(svm_scores.mean())

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print(forest_scores.mean())


plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()

input('')