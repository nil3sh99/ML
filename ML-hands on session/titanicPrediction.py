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

# x--x-x-x-x- Title x-x--x-x-x-x-x
title_mapping = {"Mr":0, "Miss":1, "Mrs":2,
                 "Master":3, "Dr":3, "Rev":3, "Mlle":3, "Major":3, "Col":3,
                 "Jonkheer":3, "Lady":3, "Mme":3, "Capt":3, "Ms":3, "Don":3,
                 "Sir":3, "Countess":3}
for dataset in train_test_data:
    dataset['Title'] = dataset["Title"].map(title_mapping)

train.head()

train.drop('Name', axis =1, inplace = True)
test.drop('Name', axis =1, inplace = True)

train.shape
test.shape

# x-x-x-x-x- SEX -x-x-x-x-x-x
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset["Sex"] = dataset["Sex"].map(sex_mapping)

train.head()

# x-x-x-x-x-x-x- AGE -x-x--x-x-x-x-x-x
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)
test["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)

train.head(30)
train.groupby("Title")

facet = sns.FacetGrid(train, hue = "Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()
plt.xlim(40,60)# change limit here

#-x-x-x--x-x Binning x-x--x-x-x-x













# x-x-x--x-x-x- Embarked x-x--x-x-x-x















#-x-x--x-x-x-x Fare x--x-x-x-xx-x--x-x

