#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from pandas import Series
import numpy as np 
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm
from statsmodels.api import OLS
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.api import qqplot  
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from pmdarima import auto_arima 
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.arima.utils import nsdiffs
from statsmodels.tsa.seasonal import seasonal_decompose 
from sklearn import datasets
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 


# In[2]:


#Importing data
Beijing = pd.read_csv("D:/Math546 Time series/final projects/PM2.5 Data of Five Chinese Cities Data Set/BeijingPM20100101_20151231.csv")


# In[3]:


#Delet non-useful data
Beijing.drop(['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'season', 'precipitation', 'Iprec'], 
               axis=1,
              inplace=True)
#Beijing.sum()


# In[4]:


Beijing.dropna(axis=0, how="any",inplace=True)


# In[5]:


# from datetime import datetime
Beijing['Time'] = pd.to_datetime(Beijing['year'].map(str) +"/"+ Beijing['month'].map(str) +"/"+ Beijing['day'].map(str)+" "+Beijing['hour'].map(str)+":00:00")
#Beijing


# In[6]:


#Beijing.groupby(['year']).size()


# In[7]:


by_hour = Beijing.groupby(['Time']).mean()
#by_hour


# In[8]:


by_hour.index = pd.to_datetime(by_hour.index, 
                             format='%d-%m-%Y %H:%M:%S')
by_hour = by_hour.set_index(by_hour.index).asfreq('H')
by_hour = by_hour.fillna(method='ffill')
#by_hour.index


# In[9]:


by_hour['Time'] = by_hour.index


# In[36]:


data_13 = by_hour[(by_hour['Time'] >=pd.to_datetime('20130101000000')) & (by_hour['Time'] <= pd.to_datetime('20140101000000'))]


# In[11]:


##len(data_13.index)


# In[12]:


data_13 = data_13.set_index(data_13.index).asfreq('H')
#data_13.index
#data_1314.index


# In[13]:


# Split data into train / test sets 
train = data_13['PM_US Post'].iloc[:len(data_13)-1000] 
test = data_13['PM_US Post'].iloc[len(data_13)-1000:]


# In[14]:



  
# Fit auto_arima function to dataset 
stepwise_fit = auto_arima(data_13['PM_US Post'], start_p = 0, start_q = 0, 
                          max_p = 2, max_q = 3, m=12,
                          start_P = 0, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise  


# In[15]:


# To print the summary 
stepwise_fit.summary() 


# In[16]:


# Fit a SARIMAX(2, 0, 0)x(1, 1, [1], 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(data_13['PM_US Post'],  
                order = (2, 0, 0),  
                seasonal_order =(1, 1, 1, 12))


# In[17]:


result = model.fit() 


# In[18]:


result.summary() 


# In[19]:


len(data_13)


# In[29]:


predict_ARMA = result.predict(start = 3600, end = 3700)

plt.figure(figsize=(12,9))
orig = plt.plot(data_13['PM_US Post'][3400:], color='blue', label='Original')
pred = plt.plot(predict_ARMA, color='red', label='Prediction' )
plt.legend(loc='best')
plt.title('Origianl vs. Prediction')
plt.show(block=False)  

