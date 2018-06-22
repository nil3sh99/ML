import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1] #reading all rows and columns except last clumn
Y = dataset.iloc[:, 3] #reading only the column which has index 3(in this case it is the last column)

np.set_printoptions(threshold = np.nan)

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) #upper bound is excluded
X[:, 1:3] = imputer.transform(X [:, 1:3])

from sklearn.preprocessing import LabelEncoder 
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:, 0])

