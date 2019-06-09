# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:04:16 2019

@author: thano
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from numpy import genfromtxt

#testlist =  np.load('bestin_129bestdist_0.00802.npy')
#
#testlist[:,1] = testlist[:,1]/testlist[:,0]
#
#
#plt.scatter(testlist[:,0], testlist[:,1])



#df = pd.read_csv('part2_stats.csv')
#
#plt.plot(df.N,df.best_dist)
#plt.xlabel('iterations')
#plt.ylabel('min distance')


df = pd.read_csv('part2_stats2.csv')

plt.plot(df.threshold,df.best_dist)
plt.xlabel('threshold')
plt.ylabel('min distance')