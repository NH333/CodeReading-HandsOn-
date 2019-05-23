import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()

#选择第3和第4的特征
X = iris["data"][:,(2,3)]
y = (iris["target"] == 2).astype(np.float64)

svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
))

svm_clf.fit(X,y)
print(svm_clf.predict([[5.5,1.7]]))

scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1,loss="hinge",random_state=42)
svm_clf2 = LinearSVC(C=100,loss="hinge",random_state=42)

svm_scaler_clf1 = Pipeline((
    ("scaler",scaler),
    ("linear_svc",svm_clf1)
))
svm_scaler_clf2 = Pipeline((
    ("scaler",scaler),
    ("linear_svc",svm_clf2),
))
svm_scaler_clf1.fit(X, y)
svm_scaler_clf2.fit(X, y)

input('')