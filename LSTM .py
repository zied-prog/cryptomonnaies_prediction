#!/usr/bin/env python
# coding: utf-8

# In[82]:


#!pip install yfinance
import yfinance as yf
import datetime

import pandas
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas_datareader as web


# In[30]:


## Lire les prix de l'Ethereum
start = datetime.datetime(2019,1,1)
end = datetime.datetime(2021,7,31)

ETH = yf.download('ETH-USD', start, end)
ETH


# In[31]:


#Creation d'une nouvelle data frame
hist_data=ETH.filter(['Close'])
hist_data.tail()


# In[32]:


#Convertion de la dataframe en numpy array
ETH_data = hist_data.values


# In[33]:


#Avoir le nombre de lignes pour entrainer le model
training_data_len = math.ceil(len(hist_data)*.8)
training_data_len


# In[34]:


# Normaliser la data
scaler = MinMaxScaler()
Scaled_data = scaler.fit_transform(hist_data)

Scaled_data,Scaled_data.shape


# In[35]:


# Creation du jeu d'entrainement

Train_data = Scaled_data[0:training_data_len,:]

# Separation de la data en jeu d'entrainement 
x_train = []
y_train = []

for i in range(60, len(Train_data)):
  x_train.append(Train_data[i-60:i,0])
  y_train.append(Train_data[i,0])
  if i<=61:
    print(x_train)
    print(y_train)
    print()
print(len(x_train))
print(len(y_train))


# In[36]:


#Convertion de x_train et y_train en numpy array
x_train, y_train = np.array(x_train), np.array(y_train)


# In[37]:


#Redimentionner la data en 3 dimentions
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[99]:


# Construction du model LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[100]:


#Compiler le model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[101]:


# Entrainer le model
model.fit(x_train, y_train, batch_size=64, epochs=500)


# In[102]:


## Creation du jeu de donnée de test

test_data = Scaled_data[training_data_len-60: , :]
#Creation de jeu de donnée x_test and y_test
x_test = []
y_test = ETH_data[training_data_len:, : ]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


# In[103]:


#Convertion de x_test en numpy array
x_test = np.array(x_test)


# In[104]:


#Redimentionner la data en 3 dimentions
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# In[105]:


### Faisons quelques prédictions
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[106]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[107]:


# Avoir les predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[108]:


print(predictions[0:10], y_test[0:10])


# In[109]:


#RMSE
rmse = np.sqrt(np.mean(predictions-y_test)**2)
rmse


# In[110]:


#MAPE
mape = np.mean(np.abs(predictions- y_test)/np.abs(y_test))  
mape


# In[111]:


#MAE
mae = np.mean(np.abs(predictions- y_test))  
mae


# In[112]:


#MPE
mpe = np.mean((predictions - y_test)/y_test)  
mpe


# In[113]:


# Plot la data
train = hist_data[:training_data_len]
valid = hist_data[training_data_len:]
valid['Predictions'] = predictions

# Visualiser la data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['True', 'Val', 'Predictions'], loc = 'lower right')
plt.show()





