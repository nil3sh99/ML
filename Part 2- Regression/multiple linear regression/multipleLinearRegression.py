#we have to  make our model predict from the independent variable to 
# predict the profit column i.e our dependent variable

# Data Pre-Processing

#importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# NOTE: In the array so formed of X,  see that th 4th column of 
# state is now replaced by the 3 columns having values as 0,1 and 
# column names being 0,1S,2
# these three values are the types of states and 0,1 in their data
# represents that for which column, which value is true


# Avoiding the dummy variable TRAP

X = X[:, 1:]



# spliting the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# pRedicting the test result

y_pred = regressor.predict(X_test)

# Backward Elimination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),      values = X,                                   axis =1)
             #adding 1's at 1st column of int type    appending values of X after 1st column        adding column put axis =0 for adding row instead
# Why do we add 1 at beginning???
# Multiple linear regression model - y = b0 + b1x1 + b2x2 ....
             # so to add the constant values we added up the 1's column values

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 