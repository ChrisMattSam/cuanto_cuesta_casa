# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:38:58 2020

@author: CSR1
"""
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from data_analysis import trim, build_and_eval
from copy import copy

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
    
    num_cols = list(df.columns)
    #merge the other cols back onto the dataframe
    drop_cols = [i for i in df.columns if i in df_first.columns]
    df_first.drop(columns = drop_cols, inplace = True)
    df = df.merge(df_first, right_index = True, left_index = True)
    df.drop(columns = ['MoSold', 'YrSold'], inplace = True)
    if numeric_cols:
        return df, num_cols
    else:
        return df

d1 =  pd.read_csv('data/train.csv').set_index('Id')
d2 = pd.concat([pd.read_csv('data/test.csv').set_index('Id'),
                    pd.read_csv('data/sample_submission.csv').set_index('Id')],1)
df, numeric_cols = continuous_vars_prep(pd.concat([d1, d2]), True)
#delete redundant features that we got rid of in the first pass:
df.drop(columns = ['TotalBsmtSF', 'TotRmsAbvGrd','YearBuilt', 'GarageYrBlt', 'GarageCars','MiscVal'],
        inplace = True)
'''
Lets bring back the ordinal categoricals, deleting features that have to do with
the date
'''

df = df.select_dtypes(exclude = ['object'])
ordinals = [i for i in df.columns if i not in numeric_cols]
print(df[ordinals].head())

'Lets start with MSSubClass, OverallCond, and OverallQual which identifies the type of dwelling involved in the sale.'
categoricals = ['MSSubClass', 'OverallQual', 'OverallCond']
for feature in categoricals:
    df[feature] = df[feature].astype('str')
    sbn.scatterplot(x = feature, y = 'SalePrice', data = df)
    plt.show()

'''
We can see the minimum SalePrice increasing as we improve in quality, but no
other immediate discernable pattern. Also isnt obvious how MSSubClass affects 
sale price. Lets check correlation with Pearson chi-square
helpful link: https://datascience.stackexchange.com/questions/893/how-to-get-correlation-between-two-categorical-variable-and-a-categorical-variab
'''

from scipy.stats import chi2_contingency
print('Chi-squared Tests')
print('Null Hypothesis: pair-wise, the compared features are independent\n')
for feature in categoricals:
    features = copy(categoricals)
    features.remove(feature)
    p = chi2_contingency(pd.crosstab(df[features[0]], df[features[1]]))[1]
    print('Features: ' + features[0] + ', ' + features[1] +
          '\np-value: ' + str(p) + '\n')

'''
From the above we see strong pair-wise correlation between these categoricals.
Lets include one of them at a time and evaluate our model:
'''


y = df['SalePrice']
X = df.drop(columns = (categoricals + ['SalePrice']) )
X = (X - X.mean())/X.std()

def rotate_in(feature, X, y):
    X = X.merge(df[feature], right_index = True, left_index = True)
    X = pd.get_dummies(X,columns = [feature], prefix = feature)
    build_and_eval(X,y,'enumerating for feature ' + feature)

for feature in categoricals:
    rotate_in(feature,X,y)

'Altogether now:'
X = X.merge(df[categoricals], right_index = True, left_index = True)
X = pd.get_dummies(X, columns = categoricals)
build_and_eval(X,y)

'''
Next steps: use the "predict" attribute of each model and plot against actual 
values to visually observe what each model is doing; pick good variables to use
as x-axis values
'''



                          