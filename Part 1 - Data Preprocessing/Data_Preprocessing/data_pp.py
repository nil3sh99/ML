import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1 ] #reading all rows and columns except last clumn
Y = dataset.iloc[:,3] #reading only the column which has index 3(in this case it is the last column)

