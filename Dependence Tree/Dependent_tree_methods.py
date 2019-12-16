# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:55:27 2019

@author: ayanca
"""


import numpy as np
import pandas as pd
from numpy.linalg import *
from math import isnan, isinf,log
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math
from scipy.stats import norm
import CycleCheck
import random

def calculate_accuracy(train,features,test,totalClass,foldSize, dimension):
    #class
    W1 = train[train[:, features] == totalClass[0]]
    #W1 = W1[:, :features]  

    W2 = train[train[:, features] == totalClass[1]]
    #W2 = W2[:, :features] 
   
    W3 = train[train[:, features] == totalClass[2]]
    #W3 = W3[:, :features] 
    
    #print (data.shape)
    W4 = train[train[:, features] == totalClass[3]]
    #W4 = W4[:, :features]
    
    trainingfeatues_X = train [:, :features]
    trainingvariables_Y = np.reshape(train[:, features],((trainingfeatues_X.shape[0]),1)) 
    
    
    weights = calculateWeight(trainingfeatues_X)
    #print(weights)    
    root, connections = getconnections(trainingfeatues_X, weights, dimension)    
    #print (root, connections)
    
    classSeparated = []
    
    testingfeatures_X = np.array(test[:,:features])
    testingvariables_Y = np.reshape(test[:, features],((foldSize),1)) 
    
    #print (trainingfeatues_X.shape, trainingvariables_Y.shape)
    #print (testingfeatures_X.shape, testingvariables_Y.shape)
    
    #print (np.size(test_X,0), np.size(test_X,1))
    #print (np.size(test_Y,0), np.size(test_Y,1))
    
    for i in range(np.size(testingfeatures_X,0)):
        classifier1 = calculateClassifier(trainingfeatues_X, trainingvariables_Y, testingfeatures_X[i,:], testingvariables_Y, weights, W1, W2, root, connections, features)
        #print "classifier -1 ", classifier1
        if classifier1 > 0.0:
            #compare class 1 and 3
            classifier2 = calculateClassifier(trainingfeatues_X, trainingvariables_Y, testingfeatures_X[i,:], testingvariables_Y, weights, W1, W3, root, connections, features)
            #print "classifier - 2", classifier2
            if classifier2 > 0.0:
                #compare class 1 and 4
                classifier3 = calculateClassifier(trainingfeatues_X, trainingvariables_Y, testingfeatures_X[i,:], testingvariables_Y, weights, W1, W4, root, connections, features)   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[0]) 
                else:
                    classSeparated.append(totalClass[3]) 
            else:
                classifier3 = calculateClassifier(trainingfeatues_X, trainingvariables_Y, testingfeatures_X[i,:], testingvariables_Y, weights, W3, W4, root, connections,features)   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])
        else:
            #compare class 2 and 3
            classifier2 = calculateClassifier(trainingfeatues_X, trainingvariables_Y, testingfeatures_X[i,:], testingvariables_Y, weights, W2, W3, root, connections, features)
            #print "classifier - 3", classifier3
            if classifier2 > 0.0:
                #compare class 2 and 4
                classifier3 = calculateClassifier(trainingfeatues_X, trainingvariables_Y, testingfeatures_X[i,:], testingvariables_Y, weights, W2, W4, root, connections, features)   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[1])  
                else:
                    classSeparated.append(totalClass[3])
            else:
                classifier3 = calculateClassifier(trainingfeatues_X, trainingvariables_Y, testingfeatures_X[i,:], testingvariables_Y, weights, W3, W4, root, connections, features)   
                if classifier3 > 0.0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])      
                
    classSeparated = np.array(classSeparated)
    #print (classSeparated)
    #print (test_Y)

    print ("The Confusion Matrix: ")
    print (confusion_matrix(testingvariables_Y, classSeparated))
    print ("The Accuracy Score: ", accuracy_score(testingvariables_Y, classSeparated)*100)
    return find_accuracy(testingvariables_Y,classSeparated)

def calculateClassifier(trainingfeatues, trainingvariables, testingfeatures, testingvariables, weights, classA, classB, root, connections, features):
   
    classValue = 0.0   
    #print (featureC1.shape)
    #print (root)
    
    RootProbabilityA = ProbCalculator(classA, root, -1, testingfeatures)
    RootProbabilityB = ProbCalculator(classB, root, -1, testingfeatures)

    classAprob = getProbabilityForMarginals(classA, connections, root, RootProbabilityA, testingfeatures) 
    classBprob = getProbabilityForMarginals(classB, connections, root, RootProbabilityB, testingfeatures) 
    
    #print(clasifierValue)
    if classAprob > classBprob:
        classValue = 1.0
    else:
        classValue = 0.0   
        
    return classValue

def find_accuracy(test_Y,test_pred):
    result = (test_Y == test_pred)
    trueDetection = (float)(np.sum(result))
    size = (int)(np.size(test_pred,0))
    #print trueDetection, size, trueDetection/size
    accuracy = (trueDetection/size)*100
    return accuracy

def calculateWeight(individualfoldfeatures):
    #print (class1features.shape) n*9
    #recieve complete training matrix
    weights = np.zeros([individualfoldfeatures.shape[1],individualfoldfeatures.shape[1]])
    depth = 1
    #print (class1features.shape[1]-1)
    for i in range(individualfoldfeatures.shape[1]-1):
        for j in range(depth,individualfoldfeatures.shape[1]):
            feature1 = individualfoldfeatures[:,i]
            feature2 = individualfoldfeatures[:,j]
            #print(feature1)
            
            feature1Set = list(set(feature1))
            feature2Set = list(set(feature2))
            
            #print(feature1Set)
            #print(feature2Set)
            
            weight = 0
            for l in feature1Set:
                for k in feature2Set:
                    count = 0
                    count_i = 0
                    count_k = 0
                    for row in range(individualfoldfeatures.shape[0]):
                        if feature1[row] == l and feature2[row] == k:
                            count += 1
                            
                        if feature1[row] == l:
                            count_i += 1
                            
                        if feature2[row] == k:
                            count_k += 1
                                            
                    p1 = count/individualfoldfeatures.shape[0]
                    p2 = count_i/individualfoldfeatures.shape[0]
                    p3 = count_k/individualfoldfeatures.shape[0]
                    if p1 == 0.0 or p2 == 0.0 or p3 == 0.0:
                        weight += 0
                    else:
                        weight += (p1) * math.log(p1/(p2*p3))
                            
            weights[i,j] = (weight)
        depth += 1
        #print (d)
    return weights

def ProbCalculator(features, root, child, sample):
    root = int(root)
    #print (root)
    child = int(child)
    if child == -1:
        top = 0
        for i in range(features.shape[0]):
            if features[i,root] == sample[root]:
                top += 1
        probability = top / features.shape[0]
    else:
        base = 0
        top = 0
        for i in range(features.shape[0]):
            if features[i,root] == sample[root]:
                base += 1
                if features[i,child] ==sample[child]:
                    top += 1
        if base == 0:
            return 0
        else:
            probability = top/base           
        
    return probability


def getProbabilityForMarginals(features, allAonnections, root, probability, sample):
    list = []
    #print('root',root)
    index = 0
    for i in range(allAonnections.shape[0]):
        if allAonnections[index, 0] == root:
            list = np.append(list, allAonnections[index, 1])
            allAonnections = np.delete(allAonnections, index, axis=0)
            index -= 1
        
        elif allAonnections[index, 1] == root:
            list = np.append(list, allAonnections[index, 0])
            allAonnections = np.delete(allAonnections, index, axis=0)
            index -= 1        
        index += 1
        
    #print(list)
    
    if any(list):
        for i in list:
            probability = probability * ProbCalculator(features, root, i, sample)
            
        for newroot in list:
            getProbabilityForMarginals(features, allAonnections, newroot, probability, sample)
                
    else:
        return probability
    return probability
            
def getconnections(trainingfeatues,weights, dimension):
    connections = []
    count = 0
    for i in range(trainingfeatues.shape[1]*trainingfeatues.shape[1]):
        link = np.unravel_index(np.argmax(weights, axis=None), weights.shape)
        #print(link, weights[link])
        a, b = link
        weights[link] = 0
        #print (weights)        
        
        if i == 0:
            connections.append(a)
            connections.append(b)
            
        else:
            g = CycleCheck.GraphCycle(dimension)
            nodes = np.asarray(connections).reshape((count+1, 2))
            for graphnode in range(int(len(connections)/2)):
                g.addEdge(nodes[graphnode,0], nodes[graphnode,1])
#                print(nodes[graphnode,0], nodes[graphnode,1])
            g.addEdge(a, b)
            if not g.isCyclic():
#                print('not cycle')
                connections.append(a)
                connections.append(b)
                count += 1                
                
        if count > dimension-2:
            continue
    
    root = random.choice(list(set(connections)))
    connections = np.asarray(connections).reshape((count+1, 2))
    return root, connections