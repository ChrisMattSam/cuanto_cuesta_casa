# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:20:38 2020

@author: christopher_sampah
"""

import pandas as pd
import numpy as np
import pylab as plt
from sklearn import linear_model as lm
from sklearn.model_selection import cross_validate as cv
import seaborn as sbn
from data_analysis import first_reg_prep

'The first iteration, with purely numerical features'
'Put the dataset together'
df = first_reg_prep(pd.read_csv('data/train.csv'))
x_train = df.drop(columns = 'SalePrice')
y_train = df.SalePrice

x_test = first_reg_prep(pd.read_csv('data/test.csv'))
y_test = pd.read_csv('data/sample_submission.csv').set_index('Id').iloc[:,0]

X = pd.concat([x_train, x_test], axis = 0)
y = pd.concat([y_train, y_test], axis = 0)

'For now lets subset the data on numerical features only'
'Exclude subsequent elements with NAs for those features'
X = X.select_dtypes(exclude = ['object']).dropna(0,'any')
y = y.loc[y.index.isin(X.index)]

model_result = cv(lm.LinearRegression(normalize = True), X, y, cv = 20,
                  scoring = 'r2', return_estimator = True)
