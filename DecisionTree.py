
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

 

# Function importing Dataset

def importdata():

    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'+'databases/balance-scale/balance-scale.data',sep= ',', header = None)

    # Printing the dataswet shape

    print ("Dataset Lenght: ", len(balance_data))

    print ("Dataset Shape: ", balance_data.shape)
     
    # Printing the dataset obseravtions

    print ("Dataset: ",balance_data.head())

    return balance_data

