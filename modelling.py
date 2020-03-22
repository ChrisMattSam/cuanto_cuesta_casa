# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:20:38 2020

@author: christopher_sampah
"""

import pandas as pd
import numpy as np
import pylab as plt
from sklearn import linear_model as lm
import seaborn as sbn

df = pd.read_csv('data/train.csv')
t = df.select_dtypes(exclude = ['object']).isnull().sum() #return numerical features only
x_train = df.loc[:, t.loc[t == 0].index].drop(columns = 'SalePrice')
y_train = df.SalePrice

df_test = pd.read_csv('data/test.csv')
x_test = df_test.loc[:, x_train.columns].dropna(axis = 0, how = 'any') # drop columns where any feature has an NA value

mod = lm.LinearRegression(normalize = True).fit(x_train.drop(columns = 'Id'), y_train)
print(mod.score(x_train.drop(columns = 'Id'), y_train)) # the R-squared (0<=R^2 <= 1), what pct of label's variability can be explained by the features

y_actual = pd.read_csv('data/sample_submission.csv')
y_actual = y_actual.loc[y_actual['Id'].isin(x_test.Id)]
y_predicted = mod.predict(x_test.drop(columns = 'Id'))
residuals = y_actual.SalePrice-y_predicted
sbn.scatterplot(data = residuals)


plt.scatter(x_test.LotArea ,y_actual.SalePrice, color = 'k')
plt.scatter(x_test.LotArea, y_predicted, color = 'g')
plt.show()



