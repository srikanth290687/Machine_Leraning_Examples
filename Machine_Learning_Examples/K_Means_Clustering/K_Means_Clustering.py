#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:51:25 2018

@author: srikanth
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/K_Means_Clustering/Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# Choosing the optimal number of cluster
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# plotting the elbow method curve to predict the number of clusters
plot.plot(range(1,11), wcss, color="red")
plot.title("The Elbow Method")
plot.xlabel("Number of cluster(s)")
plot.ylabel("WCSS")
plot.show()

# Guess the cluster based on the number of cluster obtained by the Elbow method
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)


# Visualizing the cluster results
plot.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=20, c="red", label="Carefull")
plot.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=20, c="orange", label="Standerd")
plot.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=20, c="blue", label="Target")
plot.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=20, c="green", label="Careless")
plot.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=20, c="cyan", label="Sensible")
plot.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=50, c="yellow", label="Centroid")
plot.title("Cluster of client(s)")
plot.xlabel("Annual Income")
plot.ylabel("Spending score")
plot.legend()
plot.show()