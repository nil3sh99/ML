#!/usr/bin/env python
# coding: utf-8

# # Stock Market Prediction and Forecasting using Stacked LSTM

# In[1]:


#Data Collection
import pandas_datareader as pdr


# In[3]:


key = '2ce5238ab7782b4150af996957b5b1f2f88153fb'
df = pdr.get_data_tiingo('AAPL', api_key=key)


# In[4]:


df.to_csv('AAPL.csv')


# In[5]:


import pandas as pd


# In[6]:


df = pd.read_csv('AAPL.csv')


# In[9]:


df.tail()


# In[11]:


# Picking up 'close' column
df1 = df.reset_index()['close']


# In[12]:


df1


# In[13]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[15]:


# LSTM are sensitive to the scale of data, so we apply MinMax scaler

import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[16]:


# df1 got transformed between 0-1, with a scaler factor, which is individual for every stock
df1


# In[17]:


# Splitting dataset into train and test
training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]


# In[19]:


len(train_data), len(test_data)


# In[20]:


import numpy
# Converting an array of closing values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX), numpy.array(dataY)


# In[21]:


# Reshape into x=t, t+1, t+2, t+3 and Y = t+4, the the value of time_step is 3
time_step = 100
X_train, y_train= create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[24]:


print(X_train.shape), print(y_train.shape)
# x_trian --> 100 features, y_train --> 1
# total features --> 716


# In[25]:


# Reshaping the input of X_train to a 3 Dimensional dataframe which is requirement for LSTM model
X_train= X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test= X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[26]:


print(X_train.shape), print(X_test.shape)


# In[28]:


# Creating Stacked LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[29]:


model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1))) #100,1 is the shape of x_train
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[30]:


model.summary()


# In[31]:


# Fitting the data
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)


# In[33]:


import tensorflow as tf


# In[34]:


tf.__version__


# In[35]:


# Doing the prediction and checking performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[36]:


#Transforming back to the original values from the MinMax scaler
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[42]:


# Calculating RMSE Performance Metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train, train_predict))


# In[43]:


# Test Data RMSE
math.sqrt(mean_squared_error(y_test, test_predict))


# In[44]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[47]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[48]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[49]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[50]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[51]:


import matplotlib.pyplot as plt


# In[52]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[53]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[54]:


df3=scaler.inverse_transform(df3).tolist()


# In[55]:


plt.plot(df3)


# In[ ]:




