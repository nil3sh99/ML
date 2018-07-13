#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:26:12 2018

@author: nilesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset = pd.read_csv("train.csv")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head(5)
test.head(5)

train.shape
test.shape

train.info()
test.info()