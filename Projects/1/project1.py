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
#scatter_matrix(dataset)
#plt.show()



