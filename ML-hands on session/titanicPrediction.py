#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:26:12 2018

@author: nilesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
#dataset = pd.read_csv("train.csv")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head(5)
test.head(5)

train.shape
test.shape

train.info()
test.info()

train.isnull().sum()# this tells how many data are absent for the train dataset
test.isnull().sum()# this tells how many data are absent for the test dataset

import seaborn as sns
sns.set() #setting seaborn default for plot

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead =train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['survived', 'dead']
    df.plot(kind = 'bar',stacked = True, figsize=(10,5))

bar_chart('Pclass')
bar_chart('Sex')

train_test_data = [train,test] # combining the train and the test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset["Name"].str.extract('([A-Za-z]+)\.', expand = False)

train['Title'].value_counts()











