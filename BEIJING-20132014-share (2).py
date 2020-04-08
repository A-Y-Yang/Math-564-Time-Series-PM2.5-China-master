#!/usr/bin/env python
# coding: utf-8

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


#Importing data
Beijing = pd.read_csv("/Users/kayinho/Downloads/FiveCitiePMData/BeijingPM20100101_20151231.csv")


#Delet non-useful data
Beijing.drop(['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan', 'season', 'precipitation', 'Iprec'], 
               axis=1,
              inplace=True)
#Beijing.sum()


Beijing.dropna(axis=0, how="any",inplace=True)


# from datetime import datetime
Beijing['Year.Month'] = Beijing['year'].map(str) +"/"+ Beijing['month'].map(str) 
Beijing['Date'] = pd.to_datetime(Beijing['year'].map(str) +"/"+ Beijing['month'].map(str) +"/"+ Beijing['day'].map(str))
#Beijing


#Beijing.groupby(['year']).size()


by_date = Beijing.groupby(['Date']).mean()


by_date.index = pd.to_datetime(by_date.index, 
                             format='%d-%m-%Y %H:%M')
by_date = by_date.set_index(by_date.index).asfreq('d')
by_date.index


by_date['Date'] = by_date.index


data_1314 = by_date[(by_date['Date'] >=pd.to_datetime('20130101')) & (by_date['Date'] <= pd.to_datetime('20141231'))]



data_1314 = data_1314.set_index(data_1314.index).asfreq('d')
#data_1314.index


plt.figure(figsize=(150,50)).suptitle('Daily Average PSI (2013~2014)', fontsize=150)
plt.plot(data_1314['Date'], data_1314['PM_US Post'])
plt.xlabel('Date', fontsize=120)
plt.ylabel('PM 2.5', fontsize=120)
plt.xticks(fontsize=70, rotation=0)
plt.yticks(fontsize=70, rotation=0)
plt.show()


# In[14]:


#如何确定该序列能否平稳呢？主要看：
#1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，
#ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设。
#P-value是否非常接近0.本数据中，P-value 为 2e-15,接近0.

temp = np.array(data_1314['PM_US Post'])
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


# In[15]:


#dta=pd.Series(data_1314['PM_US Post'])
#dta.index = data_1314.index
#plt.xlabel('Date')
#plt.ylabel('PM 2.5')
#dta.plot(figsize=(12,8))


# In[16]:


#fig = plt.figure(figsize=(12,8))
#ax1= fig.add_subplot(111)
#diff1 = dta.diff(1)
#diff1.plot(ax=ax1)


# In[17]:


fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(data_1314['PM_US Post'], lags=40, ax=axes[0], fft=False)
plot_pacf(data_1314['PM_US Post'], lags=40, ax=axes[1])
plt.show()


# In[18]:


res = sm.tsa.stattools.arma_order_select_ic(data_1314['PM_US Post'], ic=['aic']) 
print (res.aic_min_order)
res = sm.tsa.stattools.arma_order_select_ic(data_1314['PM_US Post'], ic=['bic']) 
print (res.bic_min_order)
res = sm.tsa.stattools.arma_order_select_ic(data_1314['PM_US Post'], ic=['hqic']) 
print (res.hqic_min_order)


# In[20]:


arma_mod03 = sm.tsa.ARMA(data_1314['PM_US Post'],(0,3)).fit()
print("ARMA(0,3): AIC=", arma_mod03.aic,", BIC= ",arma_mod03.bic,", HQ= ",arma_mod03.hqic)

arma_mod30 = sm.tsa.ARMA(data_1314['PM_US Post'],(3,0)).fit()
print("ARMA(3,0): AIC=", arma_mod30.aic,", BIC= ",arma_mod30.bic,", HQ= ",arma_mod30.hqic)

arma_mod33 = sm.tsa.ARMA(data_1314['PM_US Post'],(3,0,3)).fit()
print("ARMA(3,3): AIC=", arma_mod30.aic,", BIC= ",arma_mod30.bic,", HQ= ",arma_mod30.hqic)

arma_mod11 = sm.tsa.ARMA(data_1314['PM_US Post'],(1,1)).fit()
print("ARMA(1,1): AIC=", arma_mod33.aic,", BIC= ",arma_mod33.bic,", HQ= ",arma_mod33.hqic)

arma_mod42 = sm.tsa.ARMA(data_1314['PM_US Post'],(4,2)).fit()
print("ARMA(4,2): AIC=", arma_mod42.aic,", BIC= ",arma_mod42.bic,", HQ= ",arma_mod42.hqic)


# In[21]:


resid03 = arma_mod03.resid

fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(resid03.values.squeeze(), lags=40, ax=axes[0], fft=False)
plot_pacf(resid03, lags=40, ax=axes[1])
plt.show()


# In[22]:


resid42 = arma_mod42.resid

fig, axes = plt.subplots(1,2, figsize=(20,5))
plot_acf(resid42.values.squeeze(), lags=40, ax=axes[0], fft=False)
plot_pacf(resid42, lags=40, ax=axes[1])
plt.show()


# In[23]:


#This statistic will always be between 0 and 4. 
#The closer to 0 the statistic, the more evidence for positive serial correlation. 
#The closer to 4, the more evidence for negative serial correlation.
#the test statistic equals 2 indicates no serial correlation. 

DW_test03 = durbin_watson(resid03, axis=0)
print(DW_test03)
DW_test42 = durbin_watson(resid42, axis=0)
print(DW_test42)


# In[24]:


fig, axes = plt.subplots(1,2, figsize=(20,5))
qqplot(resid03, line='q', ax=axes[0], fit=True)
qqplot(resid42, line='q', ax=axes[1], fit=True)
plt.show()


# In[28]:


r,q,p = sm.tsa.acf(resid03.values.squeeze(), qstat=True, fft=False)
data03 = np.c_[range(1,41), r[1:], q, p]
table03 = pd.DataFrame(data03, columns=['lag', "AC", "Q", "Prob(>Q)"])


#res = arma_mod03
#sm.stats.acorr_ljungbox(res.resid, lags=[3], return_df=True)

r,q,p = sm.tsa.acf(resid42.values.squeeze(), qstat=True, fft=False)
data42 = np.c_[range(1,41), r[1:], q, p]
table42 = pd.DataFrame(data42, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table03.set_index('lag'))
print('------------------------------------------')
print(table42.set_index('lag'))


# In[31]:


predict_BEIJING03 = arma_mod03.predict('2015-01-01', '2015-12-31', dynamic=True)
print(predict_BEIJING03)

fig, ax = plt.subplots(figsize=(12, 8))
ax = data_1314['PM_US Post']['2013':].plot(ax=ax)
predict_BEIJING03.plot(ax=ax)


# In[32]:


predict_BEIJING42 = arma_mod03.predict('2015-01-01', '2015-12-31', dynamic=True)
print(predict_BEIJING42)

fig, ax = plt.subplots(figsize=(12, 8))
ax = data_1314['PM_US Post']['2013':].plot(ax=ax)
predict_BEIJING42.plot(ax=ax)


# In[43]:


predict_ARMA42 = arma_mod42.predict(start =0, end = 900)
print(predict_ARMA42)


# In[44]:


plt.figure(figsize=(12,9))
orig = plt.plot(data_1314['PM_US Post'], color='blue', label='Original')
pred = plt.plot(predict_ARMA42, color='red', label='Prediction by ARMA(0,3) Model' )
plt.legend(loc='best')
plt.title('Origianl vs. Prediction')
plt.show(block=False)               


from sklearn import linear_model
#from scipy import stats

data_1314['days_since'] = (data_1314.index - data_1314.index[0]).days

lr = linear_model.LinearRegression()
model = lr.fit(np.array([data_1314['days_since']]).reshape((-1,1)),data_1314['PM_US Post'])
rsme = model.score(np.array([data_1314['days_since']]).reshape((-1,1)),data_1314['PM_US Post'])

print('r**2 score:', rsme)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = lr.predict(np.array(data_1314['PM_US Post']).reshape((-1,1)))

plt.figure(figsize=(150,50)).suptitle('Daily Average PSI (2013~2014)', fontsize=150)
plt.plot(data_1314['Date'], data_1314['PM_US Post'])
plt.plot(data_1314['Date'], y_pred, color='blue', linewidth=3)
plt.xlabel('Date', fontsize=120)
plt.ylabel('PM 2.5', fontsize=120)
plt.xticks(fontsize=70, rotation=0)
plt.yticks(fontsize=70, rotation=0)
plt.show()

#multivariate - VAR (vector autoregression)
from statsmodels.tsa.api import VAR

temp1 = data_1314[['PM_US Post','HUMI','Iws']]
data1 = np.log(temp1).diff().dropna()
model1 = VAR(data1)

#the number is the lag order to be fit (tbc)
results = model1.fit(5)
results.summary()

results.plot()

results.plot_acorr()

model1.select_order(15)
results = model1.fit(maxlags=15, ic='aic')

lag_order = results.k_ar
results.forecast(temp1.values[-lag_order:], 5)

results.plot_forecast(10)

temp1[['PM_US Post','Iws']].plot()
plt.show()

#multivariate ARMA
temp1['const']=1

temp1['diffPM']=temp1['PM_US Post'].diff()
temp1['diffIWS']=temp1['Iws'].diff()
model2=sm.OLS(endog=temp1['diffPM'].dropna(),exog=temp1[['diffPM','const']].dropna())
results2=model2.fit()
print(results2.summary())

print(sm.tsa.stattools.grangercausalitytests(temp1[['PM_US Post','Iws']].dropna(),1))


temp1['lag']=temp1['diffIWS'].shift()
#print(temp1[['lag']])
temp1.dropna(inplace=True)

#cannot solve the following about "exog"
#model3=sm.tsa.ARIMA(endog=temp1['PM_US Post'],exog=temp1[['lag']],order=[2,0,2])
#print(model3)
#results3=model3.fit()
#print(results3.summary())