#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
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
Beijing


# In[6]:


Beijing.groupby(['year']).size()


# In[7]:


by_hour = Beijing.groupby(['Time']).mean()
by_hour


# In[8]:


by_hour.index = pd.to_datetime(by_hour.index, 
                             format='%d-%m-%Y %H:%M:%S')
by_hour = by_hour.set_index(by_hour.index).asfreq('H')
by_hour = by_hour.fillna(method='ffill')
by_hour.index


# In[9]:


by_hour['Time'] = by_hour.index


# In[10]:


data_13 = by_hour[(by_hour['Time'] >=pd.to_datetime('20130101000000')) & (by_hour['Time'] <= pd.to_datetime('20140101000000'))]


# In[11]:


data_13 = data_13.set_index(data_13.index).asfreq('H')
#data_1314.index


# In[12]:


plt.figure(figsize=(150,50)).suptitle('Daily Average PSI (2013)', fontsize=150)
plt.plot(data_13['Time'], data_13['PM_US Post'])
plt.xlabel('Time', fontsize=120)
plt.ylabel('PM 2.5', fontsize=120)
plt.xticks(fontsize=70, rotation=0)
plt.yticks(fontsize=70, rotation=0)
plt.show()


# In[13]:


#如何确定该序列能否平稳呢？主要看：
#1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，
#ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设。
#P-value是否非常接近0.本数据中，P-value 为 2e-15,接近0.

temp = np.array(data_13['PM_US Post'])
t = adfuller(temp)  # ADF test
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
output


# In[16]:


fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(data_13['PM_US Post'], lags=200, ax=axes[0], fft=False)
plot_pacf(data_13['PM_US Post'], lags=40, ax=axes[1])
plt.show()


# In[18]:


res = sm.tsa.stattools.arma_order_select_ic(data_13['PM_US Post'], ic=['aic']) 
print (res.aic_min_order)
res = sm.tsa.stattools.arma_order_select_ic(data_13['PM_US Post'], ic=['bic']) 
print (res.bic_min_order)
res = sm.tsa.stattools.arma_order_select_ic(data_13['PM_US Post'], ic=['hqic']) 
print (res.hqic_min_order)


# In[20]:


arma_mod40 = sm.tsa.ARMA(data_13['PM_US Post'],(4,0,0)).fit()
print("ARMA(4,0): AIC=", arma_mod40.aic,", BIC= ",arma_mod40.bic,", HQ= ",arma_mod40.hqic)

arma_mod20 = sm.tsa.ARMA(data_13['PM_US Post'],(2,0)).fit()
print("ARMA(2,0): AIC=", arma_mod20.aic,", BIC= ",arma_mod20.bic,", HQ= ",arma_mod20.hqic)


# In[21]:


resid40 = arma_mod40.resid

fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(resid40.values.squeeze(), lags=40, ax=axes[0], fft=False)
plot_pacf(resid40, lags=40, ax=axes[1])
plt.show()


# In[24]:


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
qqplot(resid40, line='q', ax=ax, fit=True)
plt.show()


# In[25]:


r,q,p = sm.tsa.acf(resid40.values.squeeze(), qstat=True, fft=False)
data40 = np.c_[range(1,41), r[1:], q, p]
table40 = pd.DataFrame(data40, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table40.set_index('lag'))


# In[29]:


predict_ARMA40 = arma_mod40.predict(start = 0, end = 10000)
print(predict_ARMA40)


# In[30]:


plt.figure(figsize=(12,9))
orig = plt.plot(data_13['PM_US Post'], color='blue', label='Original')
pred = plt.plot(predict_ARMA40, color='red', label='Prediction by ARMA(2,0) Model' )
plt.legend(loc='best')
plt.title('Origianl vs. Prediction')
plt.show(block=False)               


# In[ ]:




