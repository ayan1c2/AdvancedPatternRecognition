# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:53:53 2019

@author: ayanca
"""

import numpy as np
import pandas as pd
from sklearn.impute  import SimpleImputer
from glassDecisionTreeMethods import Dtree, test
from sklearn.preprocessing import Binarizer
from pprint import pprint

#Import the dataset and define the feature as well as the target datasets / columns#
cols = ['RI', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron','class']
#binarization
def get_binary(dataset):
    
    features = dataset.shape[1]
    #print (features)    
    dataset = dataset.iloc[:,:]
    datacol = np.array(dataset.iloc[1:,-1])
    datacol = np.reshape(datacol,(datacol.shape[0],1))
    dataset =  dataset.iloc[1:,:features-1] 
    #print ((np.mean(dataset,axis=0)).shape)      
    meanValue = np.mean(dataset,axis=0)       
    dataset[dataset < meanValue] = 0.0
    dataset[dataset > meanValue] = 1.0    
    #print (dataset)
    #transformer = Binarizer().fit(dataset)
    #dataset = transformer.transform(dataset) 
    dataset = np.reshape(dataset,(dataset.shape[0], dataset.shape[1]))    
    dataset = np.concatenate((dataset, datacol), axis = 1)
    #print (datacol.shape, dataset.shape)
    np.random.shuffle(dataset)
    dataset = pd.DataFrame(dataset, columns = cols)
    return dataset

#####################################################################################################################################################################
#read Data
dataset = pd.read_csv("glass.data", names = ['Index', 'RI', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron','class'])
dataset = dataset.drop('Index',axis=1)

dataset = pd.read_csv("glass.data", sep=',', low_memory=False)
dataset = dataset.iloc[:, 1:].values
dataset = pd.DataFrame(dataset, columns = cols)
data = get_binary(dataset)  
#print (dataset)
target_col_name = "class"
classValue = np.unique(data[target_col_name])
data = data.iloc[:,:].values
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
#data_test = data[:(samples-fold*foldSize),:]

################################################################################################################################################
splitArray = np.split(data[:(fold*foldSize),:], fold)
#print ("Each split size: ", splitArray[0].shape)
#print ("Total split: ", len(splitArray))

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
    #print (training_idx, training_idx.shape)   

    data_train, data_cv_test = training_idx, test_idx
    #print ("Train data Set: ", data_train)    
    #print ("CV Test data Set: ", data_cv_test.shape)   
    
    data_train = pd.DataFrame(data_train, columns = cols)
    data_cv_test = pd.DataFrame(data_cv_test, columns = cols) 
  
    #print ("For fold starts: ", (i+1))
    """
    Train the tree, Print the tree and predict the accuracy
    """
    tree = Dtree(data_train,data_train,data_train.columns[:-1])
    pprint(tree)
    accuracyVal = test(data_cv_test, tree)
    accuracy.append(accuracyVal)
    print ("For fold ends ")
   
################################################################################################################################################
print ("Average Cross Validation Accuracy for bayesian: ", sum(accuracy) / len(accuracy) )

################################################################################################################################################





