#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:36:23 2018

@author: srikanth
"""

# Simple Linear Regression

# Import the Libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the Dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Simple_Linear_Regression/Salary_Data.csv")
# Determining independent and dependent variables
X_years = dataset.iloc[:, :-1].values
y_salary = dataset.iloc[:, 1].values

# Prepare train and test set
from sklearn.cross_validation import train_test_split
X_years_train, X_years_test, y_salary_train, y_salary_test = train_test_split(X_years, y_salary, test_size=1/3, random_state=0)

# Fitting Simple Liner Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_years_train, y_salary_train)

# Predicting the Test set results
y_salary_pred = regressor.predict(X_years_test)

# Visualizing the training set results
plot.scatter(X_years_train, y_salary_train, color = 'red')
plot.scatter(X_years_test, y_salary_test, color = 'green')
plot.plot(X_years_train, regressor.predict(X_years_train), color = 'blue')
plot.title("Salary versus Experience (Training Set (Red) and Test Set(Green))")
plot.xlabel("Years of experience")
plot.ylabel("Salary")
plot.show()

# Visualizing the test set results
plot.scatter(X_years_test, y_salary_test, color = 'red')
plot.plot(X_years_train, regressor.predict(X_years_train), color = 'blue')
plot.title("Salary versus Experience (Test Set)")
plot.xlabel("Years of experience")
plot.ylabel("Salary")
plot.show()