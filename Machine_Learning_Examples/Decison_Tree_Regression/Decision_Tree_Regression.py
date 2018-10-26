#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:09:58 2018

@author: srikanth
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the dataset and seperate independent and dependent dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Polynomial Linear Regression/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fit the Decision tree regressor to the Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting the result of based on the model
y_pred = regressor.predict(6.5)

# Visualizing the results of Decision Tree Regressor
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plot.scatter(X, y, color="red")
plot.plot(X_grid, regressor.predict(X_grid), color="blue")
plot.title("Positional Level versus Salaries")
plot.xlabel("Positional Level")
plot.ylabel("Salaries")
plot.show()