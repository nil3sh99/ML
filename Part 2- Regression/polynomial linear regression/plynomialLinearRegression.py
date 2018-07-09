#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 07:40:57 2018

@author: nilesh
"""
#importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,y)

# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
# polyReg will contain x, x^2, x^3 ..... na so on

polyReg = PolynomialFeatures(degree =5)
X_poly = polyReg.fit_transform(X)
linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# Visualizing Linear Regression Model
plt.scatter(X, y, color = 'red')# represents the original points
plt.plot(X, linReg.predict(X), color = 'blue') # Represents the predicted points
plt.title("Truth OR Bluff(Linear Regression)")  
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression Model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')# represents the original points
plt.plot(X_grid, linReg2.predict(polyReg.fit_transform(X_grid)), color = 'blue') # Represents the predicted points
plt.title("Truth OR Bluff(Linear Regression)")  
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predciting a new result with Linear Regression

linReg.predict(6.5) # print the predicted salary of level 6.5

# Predictng the new result with the Polynomial Regression Model

linReg2.predict(polyReg.fit_transform(6.5))

