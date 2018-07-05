
# coding: utf-8

# In[60]:


from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
#print(iris)
print(iris.data.shape, iris.target.shape)
#both the data and target are the matrices and not the attributes themselves
X = iris.data
y = iris.target

kf = KFold(n_splits = 10)
print(kf.get_n_splits(X))
print(kf)

result = []

for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    result.append(clf.score(X_test, y_test))
    #print(clf.score(X_test, y_test))
    
print("Average accuracy is :", sum(result)/kf.get_n_splits(X))

