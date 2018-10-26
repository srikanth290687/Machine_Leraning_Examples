#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:39:59 2018

@author: srikanth
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Polynomial Linear Regression/Position_Salaries.csv")
#Extracting the independent and dependent variables
X = dataset.iloc[:, 1:2].values # Try to make the independent variable as matrics
y = dataset.iloc[:, 2].values

# No need to split the dataset into train and test as the number of records is very small

# Fit the dataset into Simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fit the dataset to Polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=4)
X_poly = pol_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the results of regressor for simple linear regression
plot.scatter(X, y, color="red")
# Including the prediction(s) of our model
plot.plot(X, lin_reg.predict(X), color = "blue")
plot.title("Positional Level v/s Salary (Liner Regression)")
plot.xlabel("Positional Level")
plot.ylabel("Salary")
plot.show()

# Visualizing the results of regressor for Ploynomial linear regression
"""X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))"""
plot.scatter(X, y, color="red")
# Including the prediction(s) of our model
plot.plot(X, lin_reg2.predict(pol_reg.fit_transform(X)), color = "blue")
plot.title("Positional Level v/s Salary (Polynomial Liner Regression)")
plot.xlabel("Positional Level")
plot.ylabel("Salary")
plot.show()

# Predicting a new result with Liner Regression
lin_reg.predict(6.5)
# Predicting a new result with Liner Regression
lin_reg2.predict(pol_reg.fit_transform(6.5))