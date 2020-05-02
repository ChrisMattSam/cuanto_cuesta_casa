# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:38:58 2020

@author: CSR1
"""
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from data_analysis import trim

def continuous_vars_prep(df, numeric_cols):
    
    '''
    A convenience fxn to bring back all the data analysis performed before the
    first modelling
    '''
    df_first = df
    
    df = trim(df,2,return_df = True)
    
    drop_cols = list(df.select_dtypes(exclude = ['number']).columns)
    [drop_cols.append(col) for col in ['MSSubClass','OverallQual','OverallCond', 'MiscVal']]

    df['LotFrontage'].replace(np.nan,0, inplace = True)
    df['MasVnrArea'].replace(np.nan,0, inplace = True)

    [drop_cols.append(i) for i in ['GarageYrBlt', 'GarageCars','TotalBsmtSF', 'TotRmsAbvGrd']]
    df.drop(columns = drop_cols, inplace = True)

    'Feature engineering and final data prep:'
    # consolidate date vars
    from operator import attrgetter
    df.rename(columns = {'YrSold': 'year','MoSold':'month'}, inplace = True)
    df['day'] = 1
    df['date_sold'] = pd.to_datetime(df[['year', 'month','day']])
    df['min_sell_date'] = df.date_sold.min()    
    df['months_since_sold'] = (df.date_sold.dt.to_period('M') - df.min_sell_date.dt.to_period('M')).apply(attrgetter('n'))
    [df.drop(columns = date_col, inplace = True) for date_col in ['year', 'month',
     'day', 'date_sold', 'min_sell_date']]
    
    df.loc[df.index.isin([2121, 2189]),['BsmtFullBath', 'BsmtHalfBath']] = 0
    df.loc[df.index == 2121, ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']] = 0
    df.loc[df['GarageArea'].isna(),['GarageArea']] = 0
    'normalize'
    df = df.sample(frac = 1) #shuffle in case the data came in an ordered manner
    
    numiadf
    #merge the other cols back onto the dataframe
    drop_cols = [i for i in df.columns if i in df_first.columns]
    df_first.drop(columns = drop_cols, inplace = True)
    df = df.merge(df_first, right_index = True, left_index = True)
    
    return df

d1 =  pd.read_csv('data/train.csv').set_index('Id')
d2 = pd.concat([pd.read_csv('data/test.csv').set_index('Id'),
                    pd.read_csv('data/sample_submission.csv').set_index('Id')],1)
df = continuous_vars_prep(pd.concat([d1, d2]))

'''
Lets bring back the ordinal categoricals
'''

df = df.select_dtypes(exclude = ['object'])




                          
                          