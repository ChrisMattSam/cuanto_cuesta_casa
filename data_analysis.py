# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:20:38 2020

@author: christopher_sampah
"""

import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sbn
from sklearn import linear_model as lm
import pickle

df = pd.read_csv('data/train.csv')
df.set_index('Id', inplace = True)
sbn.distplot(df.SalePrice/100000)

'From intro stats lets check with the 68-95-99.7 rule with respect to 1-sd, 2-sd, 3-sd'
def trim(df,std_dev=1, scale = False, return_df = False):
    mu = df.SalePrice.mean()
    sigma = df.SalePrice.std()
    if scale:
        scale = 100000
    else:
        scale = 1
    trimmed_df = df.loc[(abs(df.SalePrice) <= (mu + std_dev*sigma))]
    data = trimmed_df.SalePrice
    if return_df:
        return trimmed_df
    else:
        label = str(std_dev) + "-Sigma, " + str(round(100*data.shape[0]/df.SalePrice.shape[0],2)) + "% of dataset"
        sbn.distplot(data/scale, kde_kws = {"label": label})

sbn.distplot(df.SalePrice, kde_kws={"label": "All data"})
trim(df,1)
trim(df,2)
trim(df,3)

df = trim(df,2,return_df = True)
'find the magnitude of correlation between continuous vars with a covariance matrix'
def abs_corr(df, drop_cols = [], min_val = .6, max_val = 1):
    if len(drop_cols) > 0:
        drop_cols = list(set(drop_cols))
        df = df[[i for i in df.columns if i not in drop_cols]]
    abs_mtx = df.select_dtypes(exclude = ['object']).drop(columns = 'SalePrice').corr().abs()
    return abs_mtx[(abs_mtx < max_val) & (abs_mtx >= min_val)].dropna(1, 'all').dropna(0, 'all')

abs_corr(df)
# Lets examine further the relationship between GarageYrBlt and [YearBuilt, YearRemodAdd]
sbn.scatterplot('YearBuilt', 'GarageYrBlt',data = df)
print(df.loc[df['YearBuilt'] > df['GarageYrBlt'],['YearBuilt','GarageYrBlt']])
sbn.scatterplot('GarageYrBlt', 'YearRemodAdd',data = df)
# Because GarageYrBlt is correlated with two other features, and its relevant data still seems to be captured by the other two features, we drop it from our dataset
drop_cols = ['GarageYrBlt']
df.BsmtFullBath.value_counts()
# Next we'll get rid of feature BsmtFullBath because its correlated with BsmtFinSF1 and is less informative, and 
# because its categorical it'd require the creation of more features (e.g. dummy variables)
drop_cols.append('BsmtFullBath')

# Next we look at basement square feet and 1st floor sq feet
print(df.loc[(df['TotalBsmtSF'] == 0) & (df['1stFlrSF'] > 0)].shape)
print(df.loc[(df['TotalBsmtSF'] > 0) & (df['1stFlrSF'] == 0)].shape)
print(df.loc[df['TotalBsmtSF'] > df['1stFlrSF'],['TotalBsmtSF','1stFlrSF'] ])
print('\nLets check for houses without first floors nor basements:')
print(df.loc[df['1stFlrSF']==0].shape)
print(df.loc[df['TotalBsmtSF']==0].shape)

drop_cols.append('TotalBsmtSF')
abs_corr(df, drop_cols)

[drop_cols.append(i) for i in ['GarageCars','TotRmsAbvGrd']]
abs_corr(df, drop_cols)

# Lets take a closer look between general living area and 2nd floor square feet: 
sbn.scatterplot('GrLivArea','2ndFlrSF', data = df)

drop_cols.append('2ndFlrSF')
df.drop(columns = drop_cols, inplace = True)
df.to_pickle('data/processed_df.pickle')
