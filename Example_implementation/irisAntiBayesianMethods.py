# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:20:30 2019

@author: ayanca
"""

import numpy as np
import pandas as pd
from numpy.linalg import *
from math import isnan, isinf,log
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#import math
from statistics import mean
from scipy.stats import norm

quantile = 0.33

def calculate_accuracy(data,features,test,totalClass,foldSize):
    #mean calculation
    listClass = []
    #class
    W1 = data[data[:, features] == totalClass[0]]
    W1 = W1[:, :features]
    list = calculate_quantile(W1)    
    listClass.append(list)

    W2 = data[data[:, features] == totalClass[1]]
    W2 = W2[:, :features] 
    list = calculate_quantile(W2)    
    listClass.append(list)

    W3 = data[data[:, features] == totalClass[2]]
    W3 = W3[:, :features] 
    list = calculate_quantile(W3)    
    listClass.append(list)

    classSeparated = []
    
    test_X = np.array(test[:,:features])
    test_Y = np.reshape(test[:, features],((foldSize),1))    
    
    print (np.size(test_X,0), np.size(test_X,1))
    #print (np.size(test_Y,0), np.size(test_Y,1))
    
    for i in range(np.size(test_X,0)):
        #compare class 1 and 2
        classifier1 = calculateClassifier(test_X[i,:],listClass[0], listClass[1])
        #print "classifier -1 ", classifier1
        if classifier1 == 1:
            #compare class 1 and 3
            classifier2 = calculateClassifier(test_X[i,:],listClass[0], listClass[2])
            #print "classifier - 2", classifier2
            if classifier2 == 1:
                classSeparated.append(totalClass[0])
            else:
                classSeparated.append(totalClass[2])
        else:
            #compare class 2 and 3
            classifier3 = calculateClassifier(test_X[i,:],listClass[1], listClass[2])
            #print "classifier - 3", classifier3
            if classifier3 == 1:
                classSeparated.append(totalClass[1])
            else:
                classSeparated.append(totalClass[2])
                
    classSeparated = np.array(classSeparated)
    #print classSeparated, test_Y

    print ("The Confusion Matrix: ")
    print (confusion_matrix(test_Y, classSeparated))
    print ("The Accuracy Score: ", accuracy_score(test_Y, classSeparated)*100)
    return find_accuracy(test_Y,classSeparated)

def calculate_quantile(individualClass):
    list = []    
    #print (arr)
    
    for i in range(np.size(individualClass,1)):
        arr = np.empty([1,3])
        #print ("class-1 dimension", individualClass.shape)        
        arr [0,0], std = norm.fit(individualClass[:,i])   
        arr [0,1] = np.quantile(individualClass[:,i], quantile)
        arr [0,2] = np.quantile(individualClass[:,i], 1-quantile)
        list.append(arr)        
        #print ("array: ", arr)   
    print ("array: ", arr)
    return list

def calculateClassifier(test_X, classA, classB):
    count1 = 0; count2 = 0
    classValue = 1
    test_X = np.reshape(test_X,(1,test_X.shape[0]))
    #print (test_X.shape[0])
    for i in range(np.size(test_X,1)):
        value = test_X[:,i]
        #print ("value: ", value)
        if (value > (classA[i])[0,1]) and (value < (classA[i])[0,2]):
            count1 = count1 + 1
        elif (value > (classB[i])[0,1]) and (value < (classB[i])[0,2]):
            count2 = count2 + 1 
        elif (value > (classA[i])[0,1]) and (value < (classB[i])[0,1]):
            if (abs(value - (classA[i])[0,1])) > (abs(value - (classB[i])[0,1])):
                count2 = count2 + 1
            else: 
                count1 = count1 + 1
                
    if (count2 > count1): 
        classValue = 2
        
    return classValue

def find_accuracy(test_Y,test_pred):
    result = (test_Y == test_pred)
    trueDetection = (float)(np.sum(result))
    size = (int)(np.size(test_pred,0))
    #print trueDetection, size, trueDetection/size
    accuracy = (trueDetection/size)*100
    return accuracy