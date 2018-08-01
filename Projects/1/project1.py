#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:47:06 2018

@author: nilesh
"""
# models to be used for training dataset
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

# Load Dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv("iris.data" , names=names)

# shape of the dataset
print(dataset.shape)

# .head function will print only the argument passed values
# .describe method prints the blueprint of the dataset

#class distribution
print(dataset.groupby('class').size())

#       SINGLE VARIATE PLOTS

#box and whisker plots
dataset.plot(kind ='box', subplots = True, layout = (2,2), sharex = False, sharey= False)
plt.show()

# Drawing histograms for the input variable
dataset.hist()
plt.show()

#       MULTI VARIATE PLOTS
scatter_matrix(dataset)
plt.show()

#Splitting the dataset into training and testing datasets
X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

# Check Algos
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluating each model in turn
results =[]
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = 0)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = None)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Compare Algos using box and whisher model
fig = plt.figure()
fig.suptitle('Algo Comp')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




