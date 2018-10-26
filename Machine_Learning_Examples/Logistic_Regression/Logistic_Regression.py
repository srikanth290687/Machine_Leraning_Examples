#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:33:51 2018

@author: srikanth
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#Importing the datsset and segregating the variables
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Logistic_Regression/Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Split the dataset to train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Prediciting the test result from the model
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the result(s)
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min() - 1, stop=X_set[:,0].max() + 1, step=0.01), 
                     np.arange(start=X_set[:,1].min() - 1, stop=X_set[:,1].max() + 1, step=0.01))
plot.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red","green")))
plot.xlim(X1.min(), X1.max())
plot.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plot.scatter(X_set[y_set == j,0], X_set[y_set == j,1], c = ListedColormap(("red", "green"))(i), label=j)
plot.title("Logistic Regression (testset)")
plot.xlabel("Age")
plot.ylabel("Estimated Salaries")
plot.legend()
plot.show()