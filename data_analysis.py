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
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

def trim(df,std_dev=1, scale = False, return_df = False):
    '''
    From intro stats, check the 68-95-99.7 rule with respect to 1-sd, 2-sd,
    3-sd.
    '''
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

def abs_corr(df, drop_cols = [], min_val = .6, max_val = 1, plot_me = True, plot_x = 12, plot_y = 9):
    if len(drop_cols) > 0:
        drop_cols = list(set(drop_cols))
        df = df[[i for i in df.columns if i not in drop_cols]]
    abs_mtx = df.select_dtypes(exclude = ['object']).drop(columns = 'SalePrice').corr().abs()
    
    mtx = abs_mtx[(abs_mtx < max_val) & (abs_mtx >= min_val)].dropna(1, 'all').dropna(0, 'all')
    if plot_me:
        plt.subplots(figsize=(plot_x, plot_y))
        sbn.heatmap(mtx, vmax=.8, square=True)
    else:
        return mtx

def check_skew_kurtosis(df, feature = 'SalePrice', pics_only = False,):
    y = df[feature]
    sbn.distplot(y, fit=norm)
    plt.figure()
    stats.probplot(y, plot=plt)
    plt.show()
    print('The kurtosis: ' + str(stats.kurtosis(y)))
    print('The skew: ' + str(stats.skew(y)))
    
#if __name__ == '__main__':

df = pd.read_csv('data/train.csv')
df.set_index('Id', inplace = True)

# Initial outlier detection, check the 68-95-99.7 rule, alter the df accordingly
sbn.distplot(df.SalePrice/100000).set(xlabel = 'SalePrice (100k)')
plt.show()

sbn.distplot(df.SalePrice, kde_kws={"label": "All data"})
trim(df,1)
trim(df,2)
trim(df,3)
plt.show()

df = trim(df,2,return_df = True)

# First try with numerical features only:
abs_corr(df)
# Lets examine further the relationship between GarageYrBlt and [YearBuilt, YearRemodAdd]
sbn.scatterplot('YearBuilt', 'GarageYrBlt',data = df)
plt.show()
print(df.loc[df['YearBuilt'] > df['GarageYrBlt'],['YearBuilt','GarageYrBlt']])
sbn.scatterplot('GarageYrBlt', 'YearRemodAdd',data = df)
plt.show()
# See notebook for rationale in picking these features to drop
drop_cols = ['GarageYrBlt','TotalBsmtSF','GarageCars','TotRmsAbvGrd']
abs_corr(df, drop_cols, plot_x = 10, plot_y = 7)
sbn.scatterplot('GrLivArea','2ndFlrSF', data = df)
plt.show()
df.drop(columns = drop_cols, inplace = True)

'Feature engineering and prep'
def reg_prep(df):
    for col in df.select_dtypes(exclude = ['object']):
        if col not in ['YrSold', 'PoolArea'] and len(df[col].unique()) < 12:
            print('The col to be dummied is ' + col)
            #print(pd.get_dummies(df[col],prefix = col))
            df = pd.concat([df.drop(columns = col),pd.get_dummies(df[col],prefix = col)], axis = 1)
        return df.select_dtypes(exclude = ['object'])
y = pd.concat([df['SalePrice'].to_frame(), pd.read_csv('data/sample_submission.csv').set_index('Id')])

test = pd.read_csv('data/test.csv').set_index('Id')
df.drop(columns = 'SalePrice', inplace = True)
X = reg_prep(pd.concat([df,test[df.columns]]))
#X.isnull().sum().sort_values(ascending = False)
X.drop(columns = ['LotFrontage','MasVnrArea'], inplace = True)
X.dropna(how = 'any', inplace = True)
continuous = ['LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea',
              'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch','3SsnPorch','ScreenPorch', 'PoolArea', 'YearBuilt', 'MiscVal']

for col in continuous:
    X[col] = (X[col] - X[col].mean())/X[col].std()

dummies = [pd.get_dummies(X[categorical], prefix = categorical) for categorical in [i for i in X.columns if i not in continuous] ]
pd.concat(dummies,1)



    


