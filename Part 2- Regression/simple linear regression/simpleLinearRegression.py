#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 01:04:43 2018

@author: nilesh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #taking in all the vlaues except the last column
y = dataset.iloc[:, 1].values # fetchin the independent variable values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)

# Fitting simple linear regression model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#So what is machine learning in here??
# We have our machine as LinearRegression and then we are 
#making it learn from the training datasets of 
# X and y. Hence is application of ML