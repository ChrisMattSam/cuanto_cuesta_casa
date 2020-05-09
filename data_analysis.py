# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:20:38 2020

@author: christopher_sampah
"""

import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import sklearn
from sklearn import linear_model as lm
from sklearn.model_selection import cross_validate as cv


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
    abs_mtx = df.drop(columns = 'SalePrice').corr().abs()
    
    mtx = abs_mtx[(abs_mtx < max_val) & (abs_mtx >= min_val)].dropna(1, 'all').dropna(0, 'all')
    if plot_me:
        plt.subplots(figsize=(plot_x, plot_y))
        sbn.heatmap(mtx, vmax=.8, square=True)
        plt.show()
    else:
        return mtx

def check_skew_kurtosis(df, feature = 'SalePrice', pics_only = False,):
    '''
    Overlay a normal pdf onto the plot of the feature of interest to visually
    observe its deviation from normality
    '''
    y = df[feature]
    sbn.distplot(y, fit=norm)
    plt.figure()
    stats.probplot(y, plot=plt)
    plt.show()
    print('The kurtosis: ' + str(stats.kurtosis(y)))
    print('The skew: ' + str(stats.skew(y)))


def score_options():
    print('Possible scores to choose from: ')
    score_types = sorted(sklearn.metrics.SCORERS.keys())
    [print(s) for s in score_types]
    

def build_and_eval(X,y, extra = None, scorer = 'r2',get_max = True,
                   return_models = False, return_optimal = False,
                   score_options = False):
    '''
    Taking a (normalized) X and its corresponding y, the function builds a 
    multiple-regression model before attempting to regularize with ridge and
    lasso.
    '''
    if score_options: score_options()
    model_holder = {'Normal':[] ,'Ridge':[], 'Lasso':[]}
    baseline = cv(lm.LinearRegression(fit_intercept = True), X, y, cv = 20,
                      scoring = scorer, return_estimator = True)
    model_holder['Normal'] = baseline['estimator']
    if get_max:
        precurser = 'Largest ' + scorer + ': '
    else:
        precurser = 'Smallest ' + scorer + ': '
        
    if extra is None:
        print('Multiple Regression:')
    else:
        print('Multiple Regression ' + extra + ':')
    print(precurser +  str(baseline['test_score'].max()) + '\n')
    
    # regularize
    reg_vals = {'penalty':list(range(1,21)), 'Ridge':list(), 'Lasso':list() }
    
    for penalty in reg_vals['penalty']:
        ridger = cv(lm.Ridge(alpha = penalty), X, y, scoring = scorer,cv = 10, return_estimator = True)
        lasso = cv(lm.Lasso(alpha = penalty, max_iter = 50000), X, y, scoring = scorer,cv = 10, return_estimator = True)
        
        #obtain the min/max score and the corresponding model
        s,c = get_score_and_model(ridger['test_score'],ridger['estimator'], get_max = get_max)
        reg_vals['Ridge'].append(round(s,3))
        model_holder['Ridge'].append(c)
        
        s,c = get_score_and_model(lasso['test_score'], lasso['estimator'], get_max = get_max)
        reg_vals['Lasso'].append(round(s,3))
        model_holder['Lasso'].append(c)
        
    best_alpha = {'Ridge':0, 'Lasso':0} # use to obtain the best models based on scoring
    for val in ['Ridge', 'Lasso']:
        v = min(reg_vals[val])
        print(val + ' Regression:')
        best_alpha[val] = reg_vals['penalty'][reg_vals[val].index(v)] 
        print(precurser + str(v) + ' for corresponding alpha = ' +
              str(best_alpha[val]) + '\n')
    
    if return_optimal:
        return_models = True
        for val in ['Ridge', 'Lasso']:
            model_holder[val] = [m for m in model_holder[val] if m.alpha == best_alpha[val]]
    
    if return_models:
        return model_holder
        
def get_score_and_model(list_of_scores, list_of_models, get_max = True):
    '''
    Given a list of test scores and the corresponding list of models, 
    obtain the min/max score and its corresponding model
    '''
    if get_max:
        score_val = max(list_of_scores)
    else:
        score_val = min(list_of_scores)
        
    index_of_score = np.where(list_of_scores == score_val)[0][0] #[0][0] to get the value from the tuple
    corresponding_model = list_of_models[index_of_score]
    return score_val, corresponding_model

if __name__ == "__main__":
    'Initial outlier detection:'
    df =  pd.read_csv('data/train.csv').set_index('Id')
    df_test = pd.concat([pd.read_csv('data/test.csv').set_index('Id'),
                        pd.read_csv('data/sample_submission.csv').set_index('Id')],1)
    df = pd.concat([df, df_test])
    
    print(df.SalePrice.describe())
    
    # check the 68-95-99.7 rule, alter the df accordingly
    sbn.distplot(df.SalePrice/100000, kde_kws={"label": "Original data"})
    trim(df,1,True)
    trim(df,2,True)
    trim(df,3,True)
    plt.title('Scaled Standard Deviations')
    plt.show()
    
    'Notice the large peak. Check skew and kurtosis'
    check_skew_kurtosis(df)
    df['log_SalePrice'] = np.log(df['SalePrice'])
    check_skew_kurtosis(df,'log_SalePrice')
    # from the article (https://www.spcforexcel.com/knowledge/basic-statistics/are-skewness-and-kurtosis-useful-statistics)
    # skewness and kurtosis arent that useful, so may try regression with both logged and unlogged SalePrice
    
    # trim the data based on the graphics, come back and attempt regression without
    # removing outlier but log SalePrice
    df = trim(df,2,return_df = True)
    
    'Feature selection for modelling: continuous vars, date-vars, and counts only'
    print(df.select_dtypes(exclude = ['object']).head())
    for i in df.select_dtypes(exclude = ['object']).columns:
        print('The feature: ' + i)
        print(df[i].value_counts(dropna = False))
        print('\n')
    
    drop_cols = list(df.select_dtypes(exclude = ['number']).columns)
    [drop_cols.append(col) for col in ['MSSubClass','OverallQual','OverallCond', 'MiscVal']]
    
    'quick cleaning:'
    df['LotFrontage'].replace(np.nan,0, inplace = True)
    df['MasVnrArea'].replace(np.nan,0, inplace = True)
    
    abs_corr(df.drop(columns = 'log_SalePrice'), drop_cols)
    '''
    1st floor sq feet and basement sq feet highly correlated. Sometimes the 1st floor can be the square feet,
    and not all houses have a basement t.f. drop total bsmt sq feet.
    Also, garage cars highly correlates with garage area, and is less informative.
    Year built correlates strongly with garage's year built, and is more informative.
    Total rooms above ground highly correlates with general living area, and is less informative, and seems
    to be captured by living area as well as bedrooms above ground
    '''
    [drop_cols.append(i) for i in ['GarageYrBlt', 'GarageCars','TotalBsmtSF', 'TotRmsAbvGrd']]
    abs_corr(df.drop(columns = 'log_SalePrice'), drop_cols)
    """
    The remaining correlations arent as strong, but we note 2nd floor square feet and general living
    area. I'll leave for now since I find them both to be informative for different reasons, but 
    may drop one of them depending on model performance
    """
    df.drop(columns = drop_cols, inplace = True)
    
    'Feature engineering and final data prep:'
    # consolidate date vars
    from operator import attrgetter
    df.rename(columns = {'YrSold': 'year','MoSold':'month'}, inplace = True)
    df['day'] = 1
    df['date_sold'] = pd.to_datetime(df[['year', 'month','day']])
    df['min_sell_date'] = df.date_sold.min()
    min_sell_date = df['min_sell_date'].iloc[0]
    df['months_since_sold'] = (df.date_sold.dt.to_period('M') - df.min_sell_date.dt.to_period('M')).apply(attrgetter('n'))
    [df.drop(columns = date_col, inplace = True) for date_col in ['year', 'month',
     'day', 'date_sold', 'min_sell_date']]
    df['months_since_sold'].hist() # a quick histogram
    plt.show()
    df.corrwith(df.SalePrice).sort_values()
    
    # fill missing vals
    df.isnull().sum()
    for feature in ['BsmtFullBath','BsmtHalfBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','GarageArea']:
        print('Row of missing values for feature ' + feature+ ':')
        print(df.loc[df[feature].isna()])
        print('\n')
    
    '''
    Looking at the data, the record at indices 2121 and 2189 are missing values for all basement-related features,
    so I presume this means there doesnt exist a basement. Impute accordingly
    '''
    df.loc[df.index.isin([2121, 2189]),['BsmtFullBath', 'BsmtHalfBath']] = 0
    df.loc[df.index == 2121, ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']] = 0
    df.loc[df['GarageArea'].isna(),['GarageArea']] = 0
    df.isnull().sum()
    
    'normalize'
    df = df.sample(frac = 1) #shuffle in case the data came in an ordered manner
    X = df.drop(columns = ['SalePrice','log_SalePrice'])
    X = (X - X.mean())/X.std()
    y = df['SalePrice']
    
    
    'Modelling:'
    model_dict = build_and_eval(X,y, scorer = 'neg_mean_squared_error',
                                return_optimal = True)
    

    #lets look at the absolute residuals
    def absolute_diff(model,X,df):
        if type(model) is list:
            model = model.pop()
        y_eval = pd.Series(model.predict(X)).reindex(X.index)
        y_eval.name = 'pred_SalePrice'
        x2 = df.merge(y_eval, left_index = True, right_index = True)
        x2['diff'] = x2.SalePrice - x2.pred_SalePrice
        sbn.regplot('SalePrice','diff', data = x2)
        plt.show()
    absolute_diff(model_dict['Lasso'],X,df)
    
    '''
    Next steps: alter the fxn above to generate two plots, one for diffs and one
    for absolute diffs.  Type an explanation of potential next steps, integrate changes
    from the chi-squared tests, then MOVE ON to random forests or some other
    method.
    '''
    #[absolute_diff(l,X,df) for l in model_dict['Lasso'][:3]]
    #[absolute_diff(l,X,df) for l in model_dict['Ridge'][:3]]
    #[absolute_diff(l,X,df) for l in model_dict['Normal'][:3]]    
    
    
    