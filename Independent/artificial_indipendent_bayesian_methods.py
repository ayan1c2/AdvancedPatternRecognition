# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:34:21 2019

@author: ayanca
"""

import numpy as np
import pandas as pd
from numpy.linalg import *
from math import isnan, isinf,log
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math
from statistics import mean
from scipy.stats import norm

quantile = 0.33

def calculate_accuracy(data,features,test,totalClass,foldSize):
    #mean calculation
    listClass = []
    #class
    W1 = data[data[:, features] == totalClass[0]]
    W1 = W1[:, :features]
    list = []     
    for i in range(features):
        p1, p2 = find_probability (W1[:,i])
        #print ("Probability: ", p1, p2)
        list.append(p1)
        list.append(p2)
    listClass.append(list)    

    W2 = data[data[:, features] == totalClass[1]]
    W2 = W2[:, :features] 
    list = []     
    for i in range(features):
        p1, p2 = find_probability (W2[:,i])
        #print ("Probability: ", p1, p2)
        list.append(p1)
        list.append(p2)
    listClass.append(list)

    W3 = data[data[:, features] == totalClass[2]]
    W3 = W3[:, :features] 
    list = []     
    for i in range(features):
        p1, p2 = find_probability (W3[:,i])
        #print ("Probability: ", p1, p2)
        list.append(p1)
        list.append(p2)
    listClass.append(list)
    
    W4 = data[data[:, features] == totalClass[3]]
    W4 = W4[:, :features] 
    list = []     
    for i in range(features):
        p1, p2 = find_probability (W4[:,i])
        #print ("Probability: ", p1, p2)
        list.append(p1)
        list.append(p2)
    listClass.append(list)
    
    classSeparated = []
    
    test_X = np.array(test[:,:features])
    test_Y = np.reshape(test[:, features],((foldSize),1))    
    
    print (np.size(test_X,0), np.size(test_X,1))
    print (np.size(test_Y,0), np.size(test_Y,1))
    
    for i in range(np.size(test_X,0)):
        classifier1 = calculateClassifier(test_X[i,:],listClass[0], listClass[1], len(W1), len(W2))
        #print "classifier -1 ", classifier1
        if classifier1 > 0.0:
            #compare class 1 and 3
            classifier2 = calculateClassifier(test_X[i,:],listClass[0], listClass[2], len(W1), len(W3))
            #print "classifier - 2", classifier2
            if classifier2 > 0.0:
                #compare class 1 and 4
                classifier3 = calculateClassifier(test_X[i,:],listClass[0], listClass[3], len(W1), len(W4))   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[0]) 
                else:
                    classSeparated.append(totalClass[3]) 
            else:
                classifier3 = calculateClassifier(test_X[i,:],listClass[2], listClass[3], len(W3), len(W4))   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])
        else:
            #compare class 2 and 3
            classifier2 = calculateClassifier(test_X[i,:],listClass[1], listClass[2], len(W2), len(W3))
            #print "classifier - 3", classifier3
            if classifier2 > 0.0:
                #compare class 2 and 4
                classifier3 = calculateClassifier(test_X[i,:],listClass[1], listClass[3], len(W2), len(W4))   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[1])  
                else:
                    classSeparated.append(totalClass[3])
            else:
                classifier3 = calculateClassifier(test_X[i,:],listClass[2], listClass[3], len(W3), len(W4))   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])      
                
    classSeparated = np.array(classSeparated)
    #print (classSeparated)
    #print (test_Y)

    print ("The Confusion Matrix: ")
    print (confusion_matrix(test_Y, classSeparated))
    print ("The Accuracy Score: ", accuracy_score(test_Y, classSeparated)*100)
    return find_accuracy(test_Y,classSeparated)

def calculateClassifier(test_X, classA, classB, length1, length2):
    test_X = np.reshape(test_X,(test_X.shape[0],1))
    #print (test_X)
    #print (test_X.shape) 
    A = 0.0
    p_1 = (float) (length1) / (length1+length2)
    p_2 = (float) (length2) / (length1+length2)
    B = math.log(p_1/p_2)
    C = 0.0
    for i in range(test_X.shape[0]):
        if test_X[i,0] == 0.0:
            p_i = classA[0]
            q_i = classB[0]
            C = C + math.log(p_i/q_i)
            A = A + test_X[i,0] * ((math.log((1-p_i)/(1-q_i))) - math.log(p_i/q_i))
        else:
            p_i = classA[1]
            q_i = classB[1]
            C = C + math.log(p_i/q_i)
            A = A + test_X[i,0] * ((math.log((1-p_i)/(1-q_i))) - math.log(p_i/q_i))
    #print (A,B,C)      
    classValue = A + B + C
    return classValue

def find_accuracy(test_Y,test_pred):
    result = (test_Y == test_pred)
    trueDetection = (float)(np.sum(result))
    size = (int)(np.size(test_pred,0))    
    accuracy = (trueDetection/size)*100
    return accuracy

def find_probability(W):
    p2 = (float)(np.sum(W[W == 1.0])) / len(W)
    p1 = (1 - p2)    
    return p1, p2