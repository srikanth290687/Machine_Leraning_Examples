#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:34:09 2018

@author: srikanth
"""
# Import the Libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the Dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Simple_Linear_Regression/Salary_Data.csv")
# Determining independent and dependent variables
X_years = dataset.iloc[:, 0].values
y_salary = dataset.iloc[:, 1].values

# Since the dataset is not having any blank or NaN values, so no need to fill it with mean or median
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)"""

# Identifying the categorial data and Lables
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_years = labelencoder_X.fit_transform(X_years)
onehotencoder = OneHotEncoder(categorical_features = [0])
X_years = onehotencoder.fit_transform(X_years).toarray()
labelencoder_y = LabelEncoder()
y_salary = labelencoder_y.fit_transform(y_salary)"""

# Prepare train and test set
from sklearn.cross_validation import train_test_split
X_years_train, X_years_test, y_salary_train, y_salary_test = train_test_split(X_years, y_salary, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_years_train = sc.fit_transform(X_years_train)
X_years_test = sc.transform(X_years_test)
y_salary_train = sc.fit_transform(y_salary_train)
y_salary_test = sc.transform(y_salary_test)"""