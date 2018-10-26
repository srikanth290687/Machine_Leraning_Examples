#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 00:37:36 2018

@author: srikanth
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import random

# Import the dataset
dataset = pd.read_csv("/home/srikanth/Desktop/Machine_Learning_Examples/Reinforcement_Rule_Learning/Ads_CTR_Optimisation.csv")

# Implementing the thompson sampling
N, d = 10000, 10
ads_selected = []
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
total_rewards = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    rewards = dataset.values[n, ad]
    if rewards == 1:
        number_of_rewards_1[ad] += 1
    else:
        number_of_rewards_0[ad] += 1
    total_rewards += rewards
    
# Visualing the results
plot.hist(ads_selected)
plot.title("Histerogram of selected ads")
plot.xlabel("Ads")
plot.ylabel("Number of times each ads was selected")
plot.show()