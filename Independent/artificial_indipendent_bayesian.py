# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:32:43 2019

@author: ayanca
"""
from random import random
import numpy as np
import pandas as pd
from numpy.linalg import *
from sklearn.impute  import SimpleImputer
from sklearn import datasets
from artificial_indipendent_bayesian_methods import calculate_accuracy

################################################################################################################################################
#create data
dataSamples = 2000
d = 10
Probw1 = [random() for i in range(0, 10)]
Probw2 = [random() for i in range(0, 10)]
Probw3 = [random() for i in range(0, 10)]
Probw4 = [random() for i in range(0, 10)]
Prob = np.vstack((Probw1,Probw2,Probw3,Probw4))

data = np.zeros([dataSamples*4,d+1])

for column in range (d):
    row = 0
    for i in range (Prob.shape[0]):    
        for j in range (dataSamples):
            if random() > Prob[i,column]:
                data[row,column] = 1
                
            data[row,d] = i
            row += 1

data = pd.DataFrame(data, columns = ['Index', 'RI', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron','class'])
#print (data)
print ("Original data shape:", data.shape)
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
print ("Average Cross Validation Accuracy for Independence: ", sum(accuracy) / len(accuracy) )

################################################################################################################################################