#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 00:15:42 2018

@author: srikanth
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

# Import the dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Apriory_Association/Market_Basket_Optimisation.csv", header=None)
# Prepare the list of lists
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)