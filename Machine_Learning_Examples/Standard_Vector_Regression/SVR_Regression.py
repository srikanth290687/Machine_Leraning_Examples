#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:16:54 2018

@author: srikanth
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the dateset and extracting the independent and dependent variables
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Polynomial Linear Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Graphical represention of SVR model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plot.scatter(X, y, color="red")
plot.plot(X_grid, regressor.predict(X_grid), color="blue")
plot.title("Positional level versus Salary")
plot.xlabel("Positional Level")
plot.ylabel("Salary")
plot.show()

# Predicting the salary
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))