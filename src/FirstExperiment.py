# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:07:53 2019
Provides the sandbox just to get the first version of the code up. 
@author: lennie
"""


import pandas as pd

from zipfile import ZipFile

from urllib.request import urlopen

from io import BytesIO

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)  
url = r'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
zipped = urlopen(url)
myzip = ZipFile(BytesIO(zipped.read())).extract('hour.csv')

df = pd.read_csv(myzip)

#%%
#sns.pairplot(df[['season', 'mnth', 'hr', 'holiday', 'weekday',
#       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
#       'casual', 'registered', 'cnt']].sample(1000))
#plt.tight_layout()
enc = pd.DataFrame(OneHotEncoder(sparse=False).fit_transform(df['weekday'].values.reshape(-1,1)))
df = pd.concat([df, enc],axis=1)
#%%
#df = df
X_Keys = ['mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'hum', 'windspeed',0,1,2,3,4,5]
y_Keys = 'cnt'
#%%
X_train, X_test, y_train, y_test = train_test_split(
    df[X_Keys].astype("float"), df[y_Keys].astype("float"), test_size=0.33, random_state=42)

#
#print("hel")
#model = xgb.XGBRegressor(max_depth=7)
#
#model.fit(X_train.values,y_train.values)
#print(model.score(X_train.values,y_train.values))
#print(model.score(X_test.values,y_test.values))
#residuals = model.predict(X_test.values) - y_test.values
#
#sample1 = df['cnt'].sample(1000,replace=True).values
#sample2 = df['cnt'].sample(1000,replace=True).values
#diffs = abs(sample1 - sample2)
#plt.hist(diffs,density=True)
#plt.hist(abs(residuals),density=True)
#print("hel")

#%%

parameters = {'reg_lambda':[0.1,0.3,1],'reg_alpha':[0.1,0.3,1],'n_estimators':[100],'gamma':[0.1,0.2,0.9],'max_depth':[6,7,8,9,10,11,12,13,14],'learning_rate':[0.01,0.1,0.3,0.5]}

clf = xgb.XGBRegressor(base_score = 2,n_jobs=-1)
search = GridSearchCV(clf,parameters,cv=3)
search.fit(X_train,y_train)


#%%
print(search.best_estimator_.score(X_train,y_train))
print(search.best_estimator_.score(X_test,y_test))
residuals = search.best_estimator_.predict(X_test) - y_test.values
mae = mean_absolute_error(search.best_estimator_.predict(X_test), y_test.values)

sample1 = df['cnt'].sample(1000,replace=True).values
sample2 = df['cnt'].sample(1000,replace=True).values
mae2 = mean_absolute_error(sample1,sample2)
diffs = abs(sample1 - sample2)
#%%
import numpy as np 
store = []
for i in range(1000):
    store.append(np.random.choice(residuals)> np.random.choice(diffs))

#%%
#plt.hist(abs(diffs),density=True,bins=50)
plt.hist(abs(residuals),density=True,histtype='step', cumulative=-1,bins=250)
