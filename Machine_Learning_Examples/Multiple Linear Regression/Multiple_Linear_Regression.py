#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:28:32 2018

@author: srikanth
"""

# Import the datadet 
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Multiple Linear Regression/50_Startups.csv")
# Sepreating the indendent and dependent variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Identifying the categorial variable(s)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lebelencoder_X = LabelEncoder()
X[:, 3] = lebelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Preparing Train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state=0)

# Fitting our multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elemination Model

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()