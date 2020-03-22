# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:20:38 2020

@author: christopher_sampah
"""

import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sbn

df = pd.read_csv('data/train.csv')
sbn.distplot(df.SalePrice/100000)

'From intro stats lets check with the 68-95-99.7 rule with respect to 1-sd, 2-sd, 3-sd'
def trim(df,std_dev=1, scale = False ):
    mu = df.SalePrice.mean()
    sigma = df.SalePrice.std()
    if scale:
        scale = 100000
    else:
        scale = 1
    data = df.loc[(abs(df.SalePrice) <= (mu + std_dev*sigma))].SalePrice
    
    label = str(std_dev) + "-Sigma, " + str(round(100*data.shape[0]/df.SalePrice.shape[0],2)) + "% of dataset"
    sbn.distplot(data/scale, kde_kws = {"label": label})

sbn.distplot(df.SalePrice, kde_kws={"label": "All data"})
trim(df,1)
trim(df,2)
trim(df,3)

"lets be conservative and take data within 3-sigma for now, but we'll adjust accordingly as needed later"


