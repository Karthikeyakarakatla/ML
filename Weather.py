# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 15:24:35 2022

@author: lenovo
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/lenovo/Desktop/Weather_data.csv",skipinitialspace=True)
df.head()

df.shape
df.info()

df.isnull().sum()

df = df.dropna(axis = 'columns', how = 'all')
df

df.shape

half_count = len(df)/2
df = df.dropna(thresh=half_count,axis=1)
df.head()

print(df.shape)
df.isnull().sum()

df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
df = df.set_index('datetime_utc')
df.head()

# rename columns
new_cols = [x.replace('_','') for x in df.columns]
df.columns = new_cols

df.columns
df.fillna(method='ffill',inplace=True)
df.isnull().sum()

df["conds"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
columns=(['conds','wdire'])
for col in columns:
    df[col] = le.fit_transform(df[col])
    
df.head()

#Check Correlation between Columns 
correlation = df.corr() 
fig, ax = plt.subplots(figsize=(16,10)) 
ax = sns.heatmap(correlation ,annot = True)

print("most important features relative to target")
correlation.sort_values(['tempm'], ascending=False, inplace=True)
correlation.tempm

X=df.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14]].values
Y=df.iloc[:,8].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

#Checking the accuracy
lr.score(X,Y)

#predict the test set results
y_pred = lr.predict(X)
y_pred

from sklearn.metrics import mean_squared_error
from math import sqrt

mean_squared_error(Y,y_pred)

rmse = sqrt(mean_squared_error(Y, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(Y, y_pred)

adj_r2 = 1 - float(len(Y)-1)/(len(Y)-len(lr.coef_)-1)*(1 - r2)

rmse, r2, adj_r2, lr.coef_, lr.intercept_

# temperature Frequency
plt.figure(figsize=(20, 10));
df.tempm.value_counts().head(50).plot(kind='bar');
plt.title("Weather Conditions in Delhi")
plt.plot();
# most common temp is 29 degree C

