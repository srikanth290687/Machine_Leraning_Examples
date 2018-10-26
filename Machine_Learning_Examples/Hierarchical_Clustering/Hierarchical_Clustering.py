#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 00:06:10 2018

@author: srikanth
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/K_Means_Clustering/Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# Determining the number of clusters by drawing the dendrogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plot.title("Dendrogram")
plot.xlabel("Customer")
plot.ylabel("Euclidean Distance")
plot.show()

# Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizing the results
plot.scatter(X[y_hc==0,0], X[y_hc==0,1], s=15, c="blue", label="Carefull")
plot.scatter(X[y_hc==1,0], X[y_hc==1,1], s=15, c="green", label="Standerd")
plot.scatter(X[y_hc==2,0], X[y_hc==2,1], s=15, c="yellow", label="Target")
plot.scatter(X[y_hc==3,0], X[y_hc==3,1], s=15, c="orange", label="Careless")
plot.scatter(X[y_hc==4,0], X[y_hc==4,1], s=15, c="red", label="Sensible")
plot.title("Hierarchical Clustering")
plot.xlabel("Annual Income")
plot.ylabel("Spending Score")
plot.legend()
plot.show()