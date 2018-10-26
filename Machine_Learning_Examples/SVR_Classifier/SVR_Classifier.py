#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:08:08 2018

@author: srikanth
"""
# Import the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the dataset and segregarte the independent and dependent variables
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Logistic_Regression/Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Split the dataset to test and train set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit the SVM Classifier to train set
from sklearn.svm import SVC
classifier = SVC(kernel="linear", random_state=0)
classifier.fit(X_train, y_train)

# Predicting the result
y_pred = classifier.predict(X_test)

# Calculating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the result (for train set)
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01), 
                    np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
plot.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), 
              alpha=0.60, cmap=ListedColormap(("red","green")))
plot.xlim(X1.min(),X1.max())
plot.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plot.scatter(X_set[y_set==j,0], X_set[y_set==j,1], c=ListedColormap(("red","green"))(i), label=j)
plot.title("SVR Classifier (Train Set))")
plot.xlabel("Age")
plot.ylabel("Estimated Salary")
plot.legend()
plot.show()

# Visualizing the result (for test set)
X_set, y_set = X_test, y_test
X1, X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1, step=0.01), 
                  np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
plot.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), 
              alpha=0.60, cmap=ListedColormap(("red","green")))
plot.xlim(X1.min(),X1.max())
plot.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plot.scatter(X_set[y_set==j,0], X_set[y_set==j,1], c=ListedColormap(("red","green"))(i), label=j)
plot.title("SVR Classifier (Test Set))")
plot.xlabel("Age")
plot.ylabel("Estimated Salary")
plot.legend()
plot.show()