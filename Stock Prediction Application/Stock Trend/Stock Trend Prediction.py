#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[68]:


#defining starting and ending date for the application
start = '2009-12-31'
end = '2019-12-31' 

# Here the user will enter the stock ticker
df = data.DataReader('AAPL', 'yahoo', start, end)
df.head()


# In[69]:


df.tail()


# In[70]:


df = df.reset_index()
df.head()


# In[71]:


# Removing the headers which are not so useful in our analysis
df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()


# In[72]:


#Shows the closing point of the AAPL stock
plt.plot(df.Close)
df


# In[73]:


# Calculating the moving average of the company
ma100 = df.Close.rolling(100).mean()
ma100


# In[74]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')


# In[75]:


# Calculating the moving average for 200 days
ma200 = df.Close.rolling(200).mean()
ma200

#Plotting the moving average for 200 days and 100 days together
plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')


# In[76]:


df.shape


# In[77]:


# Splitting Data into training (70%) and testing (30%)
# We have chosen the CLOSE column from the Yahoo Data because we are interested into that only
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)


# In[78]:


data_training.head()


# In[79]:


data_testing.head()


# In[80]:


# Scaling down the data between 0 and 1 for the LSTM Model

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# In[81]:


# Fitting the training data into the feature range defined above for LSTM Model --> Training data
# We have scaled down the training data and converted that into an array called data_training_array

data_training_array = scaler.fit_transform(data_training)
data_training_array


# In[82]:


# 1761 rows are in the training part and the rest are in the testing part
data_training_array.shape


# In[83]:


# Defining two lists for prediction 

x_train = []
y_train = []

# step size = 100
# append(i-100), because we want to start our x_train array from 0-100 and y_train array from the next value which
# which is controlled by our loop variable 'i'
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train.shape # We see 100 as the number of columns which are acting as our features for the model 


# In[84]:


# ML Model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


# In[85]:


model = Sequential()

#Layer 1
model.add(LSTM(units=50, activation = 'relu', return_sequences = True, 
              input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

#Layer 2
model.add(LSTM(units=60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

#Layer 3
model.add(LSTM(units=80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

#Layer 4
model.add(LSTM(units=120, activation = 'relu'))
model.add(Dropout(0.5))

          
#Layer 5 and only one value which is the Closing Price
model.add(Dense(units = 1))


# In[86]:


model.summary()


# In[113]:


model.compile(optimizer='adam', loss = 'mean_squared_error')
# We chose Mean Squared Error because we are doing a time-domain analysis, other types of losses are usually used for the 
# classification purposes

model.fit(x_train, y_train, epochs = 50)


# In[114]:


model.save('keras_model.h5')


# In[115]:


# to predict the values in testing data from the training data, we have to fetch the last 100 days data from the training
# set which can be obtained using data_training.tail() method
past_100_days = data_training.tail(100)


# In[116]:


final_df = past_100_days.append(data_testing, ignore_index=True)


# In[117]:


final_df.head()


# In[118]:


input_data = scaler.fit_transform(final_df)
input_data


# In[119]:


input_data.shape


# In[120]:


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])


# In[121]:


x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[122]:


# Making predicitons
y_predicted = model.predict(x_test)


# In[123]:


y_predicted.shape


# In[124]:


y_test


# In[125]:


scaler.scale_


# In[126]:


y_predicted


# In[127]:


scale_factor = 1/0.02099517
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# In[128]:


plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




