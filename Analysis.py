# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:14:07 2019

@author: ayanca
"""

import numpy as np
import pandas as pd

data = pd.read_csv("data/glass2.data", sep=',', low_memory=False)
#print "Original data:", data
print ("Original data shape:", data.shape)
#print data

data = data.iloc[:,1:].values

features = np.size(data,1)-1 # 149  #column    [all columns except last one as it has predicted class]
#print features
samples = np.size(data,0)  #row

totalClass = data[:, features]
totalClass = (np.sort(np.unique(np.array(totalClass))))
totalClass = np.reshape(totalClass,[totalClass.size,1]) #convert to (*,1) array
print ("classes: ", totalClass.shape[0])

for i in range(totalClass.shape[0]):
    W1 = data[data[:, features] == totalClass[i]]
    print (totalClass[i], W1.shape)