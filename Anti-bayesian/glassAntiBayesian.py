# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:00:50 2019

@author: ayanca
"""

import numpy as np
import pandas as pd
from numpy.linalg import *
from sklearn.impute  import SimpleImputer
from sklearn import datasets
from glassAntiBayesianMethods import calculate_accuracy

################################################################################################################################################
#load data
data = pd.read_csv("data/glass.data", sep=',', low_memory=False)
#print ("Original data:", data)
print ("Original data shape:", data.shape)
#print data
np.seterr(divide = 'ignore') 

################################################################################################################################################
#check if missing value (cleaning)
missing_values = ["n/a", "na", "--","?", " ","NA"]
data = data.replace(missing_values, np.nan)
feat_miss = data.columns[data.isnull().any()]
if feat_miss.size == 0:
    print ("Data is clean")
else:
    print ("Missing data shape before:", feat_miss.shape)
    imputer = SimpleImputer(copy=True, fill_value=None, missing_values=np.nan, strategy='calculate_iris', verbose=0)
    data[feat_miss] = imputer.fit_transform(data[feat_miss])
    feat_miss = data.columns[data.isnull().any()]
    print ("Missing data shape after:", feat_miss.shape)

################################################################################################################################################
#shuffle data
data = data.sample(frac=1) #Returns a random sample of items from an axis of object.
print ("Shape after shuffling", data.shape)

################################################################################################################################################
#removed head and index and convert to numpy array
data = data.iloc[:, 1:].values  #np.random.shuffle(data)     print data.size
#print "Cleaned data:", data
print ("Cleaned data shape:", data.shape)

################################################################################################################################################
#data Training
features = np.size(data,1)-1 #column    [all columns except last one as it has predicted class]
samples = np.size(data,0)  #row
#print (features, samples)

################################################################################################################################################
#class finding
totalClass = data[:, features]
totalClass = (np.sort(np.unique(np.array(totalClass))))
totalClass = np.reshape(totalClass,[totalClass.size,1]) #convert to (*,1) array
print ("classes: ", totalClass)

################################################################################################################################################
fold = 5
foldSize = (int)(((float)(samples)/fold))
print ("fold Size: ", foldSize)

################################################################################################################################################
#pure test data
data_test = data[:(samples-fold*foldSize),:]

################################################################################################################################################
splitArray = np.split(data[:(fold*foldSize),:], fold)
print ("Each split size: ", splitArray[0].shape)
print ("Total split: ", len(splitArray))

################################################################################################################################################

#call method
print ("For fold starts for optimal bayesian")
accuracy = []
for i in range(fold-1):    
    
    training_idx = []    
    
    test_idx = splitArray[i]
    for j in range(len(splitArray)):
        if j !=i:
            training_idx.append(splitArray[i])
            
    training_idx = np.array(np.concatenate((training_idx), axis=0))
    #print training_idx, training_idx.shape   

    data_train, data_cv_test = training_idx, test_idx
    #print "Train data Set: ", data_train.shape    
    #print "CV Test data Set: ", data_cv_test.shape   
  
    print ("For fold starts: ", (i+1))
    accuracyVal = calculate_accuracy(data_train,features,data_cv_test,totalClass,foldSize)
    accuracy.append(accuracyVal)
    print ("For fold ends ")
   
################################################################################################################################################
print ("Average Cross Validation Accuracy for Anti-bayesian: ", sum(accuracy) / len(accuracy) )

################################################################################################################################################