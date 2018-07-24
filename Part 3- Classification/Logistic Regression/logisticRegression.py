'''
@author: nilesh
'''
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Processing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# Splitting the data set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting LR to the training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

# Confusion matrix gives the numner of correct and incorrect predictions
# Starting from the top left and going along the diagnol gives us the 
# correct values and vice versa

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
