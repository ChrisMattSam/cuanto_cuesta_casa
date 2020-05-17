Author: Christopher Sampah
Email: christopher.m.sampah@gmail.com
Date: March-April 2020

This repo contains all files for my attempt at a kaggle competition found here: https://www.kaggle.com/c/house-prices-advanced-regression-techniques. The goal is to utilize multivariate regression before implementing regularizers ridge and lasso.  In this approach I am VERY averse to using models that aren't easily understood at the expense of model accuracy (e.g. I could use xgboost and call it a day but chose not to).

I lightly referenced Kaggle notebook https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python by Pedro Marcelino to get started with my data analysis, primarily using the correlation matrix idea and further examined skewness & kurtosis due to him referencing it.

The general idea is to build a model that can be easily explained and understood to a lay-person, hence why I start with purely continuous features as a baseline and gradually progress from there. I don't initially go into more advanced modelling mentioned by two other very popular Kaggle notebooks I read through (https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard by Serigne and https://www.kaggle.com/apapiu/regularized-linear-models by Alexandru Papiu) for this purpose.
