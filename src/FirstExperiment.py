# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:07:53 2019

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

#%%
df = df.head(1000)
X_Keys = ['season', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered']
y_Keys = 'cnt'

X_train, X_test, y_train, y_test = train_test_split(
    df[X_Keys].astype("float"), df[y_Keys].astype("float"), test_size=0.33, random_state=42)

print("hel")
model = xgb.XGBRegressor()

model.fit(X_train.values,y_train.values)

print("hel")