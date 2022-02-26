# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:00:33 2022

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

df = pd.read_csv("C:/Users/lenovo/Desktop/ailerons_train.csv")
df.head()

df.shape
df.isnull().sum()

#Check Correlation between Columns 
correlation = df.corr() 
fig, ax = plt.subplots(figsize=(16,10)) 
ax = sns.heatmap(correlation ,annot = True)

plt.figure(figsize=(9, 8))
sns.distplot(df['goal'], color='g', bins=100, hist_kws={'alpha': 0.4});

X=df.iloc[:,:39].values
Y=df.iloc[:,40].values
Y

#split dataset into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Apply various regression models and find out which model is the best for this dataset
#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#predict the test set results
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(" rmse value of Linear Regression is : ",rmse)
r2 = r2_score(y_test, y_pred)
print(r2)

rmse value of Linear Regression is :  0.00017488070679901087
0.8038654033509506