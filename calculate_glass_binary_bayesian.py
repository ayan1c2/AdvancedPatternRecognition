import numpy as np
#import pandas as pd
from numpy.linalg import *
from math import isnan, isinf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#import math

def calculate_accuracy(data,features,test,totalClass,foldSize):
    #mean calculation
    #class
    W1 = data[data[:, features] == totalClass[0]]
    W1 = W1[:, :features] 
    #print "class-1 dimension", W1.shape
    mean_class1 = mean_calulate(W1)
    print ("Class-1 mean: ", mean_class1.shape)
    W1_cov = find_cov(W1,mean_class1,features)
    print ("COV class-1: ", W1_cov.shape)
    #print np.isnan(W1_cov)
    W1_cov_inv = pinv(W1_cov)
    print ("Inverse COV class-1: ", W1_cov_inv.shape)

    W2 = data[data[:, features] == totalClass[1]]
    W2 = W2[:, :features] 
    print ("class-2 dimension", W2.shape)
    mean_class2 = mean_calulate(W2)
    print ("Class-2 mean: ", mean_class2.shape)
    W2_cov = find_cov(W2,mean_class2,features)
    print ("COV class-2: ", W2_cov.shape)
    W2_cov_inv = pinv(W2_cov)
    print ("Inverse COV class-2: ", W2_cov_inv.shape)

    W3 = data[data[:, features] == totalClass[2]]
    W3 = W3[:, :features] 
    #print "class-3 dimension", W3.shape
    mean_class3 = mean_calulate(W3)
    print ("Class-3 mean: ", mean_class3.shape)
    W3_cov = find_cov(W3,mean_class3,features)
    print ("COV class-3: ", W3_cov.shape)
    W3_cov_inv = pinv(W3_cov)
    print ("Inverse COV class-3: ", W3_cov_inv.shape)
    
    W4 = data[data[:, features] == totalClass[3]]
    W4 = W4[:, :features] 
    #print "class-4 dimension", W4.shape
    mean_class4 = mean_calulate(W4)
    print ("Class-4 mean: ", mean_class4.shape)
    W4_cov = find_cov(W4,mean_class4,features)
    print ("COV class-4: ", W4_cov.shape)
    W4_cov_inv = pinv(W4_cov)
    print ("Inverse COV class-4: ", W4_cov_inv.shape)
           
    '''
    W5 = data[data[:, features] == totalClass[4]]
    W5 = W5[:, :features] 
    #print "class-5 dimension", W5.shape
    mean_class5 = mean_calulate(W5)
    print "Class-5 mean: ", mean_class5.shape
    W5_cov = find_cov(W5,mean_class5,features)
    print "COV class-5: ", W5_cov.shape
    W5_cov_inv = pinv(W5_cov)
    print "Inverse COV class-5: ", W5_cov_inv.shape    
    
    W6 = data[data[:, features] == totalClass[5]]
    W6 = W6[:, :features] 
    #print "class-6 dimension", W6.shape
    mean_class6 = mean_calulate(W6)
    print "Class-6 mean: ", mean_class6.shape
    W6_cov = find_cov(W6,mean_class6,features)
    print "COV class-6: ", W6_cov.shape
    W6_cov_inv = pinv(W6_cov)
    print "Inverse COV class-6: ", W6_cov_inv.shape
    '''
    classSeparated = []
    #print test.shape

    test_X = np.transpose(test[:,:features])
    print (test_X.shape)
    test_Y = np.reshape(test[:, features],((foldSize),1))

    for i in range(np.size(test_X,1)):
        #compare class 1 and 2
        classifier1 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W1_cov_inv,W2_cov_inv,W1_cov,W2_cov,mean_class1,mean_class2)
        #print "classifier -1 ", classifier1
        if classifier1 > 0:
            #compare class 1 and 3
            classifier2 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W1_cov_inv,W3_cov_inv,W1_cov,W3_cov,mean_class1,mean_class3)
            #print "classifier - 2", classifier2
            if classifier2 > 0:
                #compare class 1 and 4
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W1_cov_inv,W4_cov_inv,W1_cov,W4_cov,mean_class1,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[0]) 
                else:
                    classSeparated.append(totalClass[3]) 
            else:
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W3_cov_inv,W4_cov_inv,W3_cov,W4_cov,mean_class3,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])
        else:
            #compare class 2 and 3
            classifier2 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W2_cov_inv,W3_cov_inv,W2_cov,W3_cov,mean_class2,mean_class3)
            #print "classifier - 3", classifier3
            if classifier2 > 0:
                #compare class 2 and 4
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W2_cov_inv,W4_cov_inv,W2_cov,W4_cov,mean_class2,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[1])  
                else:
                    classSeparated.append(totalClass[3])
            else:
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W3_cov_inv,W4_cov_inv,W3_cov,W4_cov,mean_class3,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])  
            
    classSeparated = np.array(classSeparated)
    #print classSeparated, test_Y

    print ("The Confusion Matrix: ")
    print (confusion_matrix(test_Y, classSeparated))
    print ("The Accuracy Score: ", accuracy_score(test_Y, classSeparated)*100)
    return find_accuracy(test_Y,classSeparated)

def calculate_accuracy_naive(data,features,test,totalClass,foldSize):
    #mean calculation
    #class
    W1 = data[data[:, features] == totalClass[0]]
    W1 = W1[:, :features] 
    #print "class-1 dimension", W1.shape
    mean_class1 = mean_calulate(W1)
    print ("Class-1 mean: ", mean_class1.shape)
    W1_cov = find_cov_naive(W1,mean_class1,features)    
    print ("COV naive class-1: ", W1_cov.shape)
    #print np.isnan(W1_cov)
    W1_cov_inv = pinv(W1_cov)
    print ("Inverse COV class-1: ", W1_cov_inv.shape)

    W2 = data[data[:, features] == totalClass[1]]
    W2 = W2[:, :features] 
    #print "class-2 dimension", W2.shape
    mean_class2 = mean_calulate(W2)
    print ("Class-2 mean: ", mean_class2.shape)
    W2_cov = find_cov_naive(W2,mean_class2,features)
    print ("COV class-2: ", W2_cov.shape)
    W2_cov_inv = pinv(W2_cov)
    print ("Inverse COV class-2: ", W2_cov_inv.shape)

    W3 = data[data[:, features] == totalClass[2]]
    W3 = W3[:, :features] 
    #print "class-3 dimension", W3.shape
    mean_class3 = mean_calulate(W3)
    print ("Class-3 mean: ", mean_class3.shape)
    W3_cov = find_cov_naive(W3,mean_class3,features)
    print ("COV class-3: ", W3_cov.shape)
    W3_cov_inv = pinv(W3_cov)
    print ("Inverse COV class-3: ", W3_cov_inv.shape)
    
    W4 = data[data[:, features] == totalClass[3]]
    W4 = W4[:, :features] 
    #print "class-4 dimension", W4.shape
    mean_class4 = mean_calulate(W4)
    print ("Class-4 mean: ", mean_class4.shape)
    W4_cov = find_cov(W4,mean_class4,features)
    print ("COV class-4: ", W4_cov.shape)
    W4_cov_inv = pinv(W4_cov)
    print ("Inverse COV class-4: ", W4_cov_inv.shape)
    
    '''
    W5 = data[data[:, features] == totalClass[4]]
    W5 = W5[:, :features] 
    #print "class-5 dimension", W5.shape
    mean_class5 = mean_calulate(W5)
    print "Class-5 mean: ", mean_class5.shape
    W5_cov = find_cov(W5,mean_class5,features)
    print "COV class-5: ", W5_cov.shape
    W5_cov_inv = pinv(W5_cov)
    print "Inverse COV class-5: ", W5_cov_inv.shape
    
    W6 = data[data[:, features] == totalClass[5]]
    W6 = W6[:, :features] 
    #print "class-6 dimension", W6.shape
    mean_class6 = mean_calulate(W6)
    print "Class-6 mean: ", mean_class6.shape
    W6_cov = find_cov(W6,mean_class6,features)
    print "COV class-6: ", W6_cov.shape
    W6_cov_inv = pinv(W6_cov)
    print "Inverse COV class-6: ", W6_cov_inv.shape
    '''
    classSeparated = []

    test_X = np.transpose(test[:,:9])
    #print test_X.shape
    test_Y = np.reshape(test[:, 9],((foldSize),1))
    #print np.size(test_X,1)
    
    for i in range(np.size(test_X,1)):
        #compare class 1 and 2
        classifier1 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W1_cov_inv,W2_cov_inv,W1_cov,W2_cov,mean_class1,mean_class2)
        #print "classifier -1 ", classifier1
        if classifier1 > 0:
            #compare class 1 and 3
            classifier2 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W1_cov_inv,W3_cov_inv,W1_cov,W3_cov,mean_class1,mean_class3)
            #print "classifier - 2", classifier2
            if classifier2 > 0:
                #compare class 1 and 4
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W1_cov_inv,W4_cov_inv,W1_cov,W4_cov,mean_class1,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[0]) 
                else:
                    classSeparated.append(totalClass[3]) 
            else:
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W3_cov_inv,W4_cov_inv,W3_cov,W4_cov,mean_class3,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])
        else:
            #compare class 2 and 3
            classifier2 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W2_cov_inv,W3_cov_inv,W2_cov,W3_cov,mean_class2,mean_class3)
            #print "classifier - 3", classifier3
            if classifier2 > 0:
                #compare class 2 and 4
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W2_cov_inv,W4_cov_inv,W2_cov,W4_cov,mean_class2,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[1])  
                else:
                    classSeparated.append(totalClass[3])
            else:
                classifier3 = calculateClassifier(np.reshape(test_X[:,i],(9,1)),W3_cov_inv,W4_cov_inv,W3_cov,W4_cov,mean_class3,mean_class4)   
                if classifier3 > 0:
                    classSeparated.append(totalClass[2]) 
                else:
                    classSeparated.append(totalClass[3])  
                               
    classSeparated = np.array(classSeparated)
    #print classSeparated, test_Y
           
    print ("The Confusion Matrix: ")
    print (confusion_matrix(test_Y, classSeparated))
    print ("The Accuracy Score: ", accuracy_score(test_Y, classSeparated)*100)
    return find_accuracy(test_Y,classSeparated)

def mean_calulate(X):
    N = np.size(X,0) #0: total row
    M = np.size(X,1) #1: total col
    #print X
    summation = np.sum(X, axis=0) #0: row wise sum:  np.sum([[0, 1], [0, 5]], axis=0) = array([0, 6])
    #print summation, N
    mean = summation / N 
    return np.reshape(mean,(M,1))

def find_cov(W1_X, W1_mean,features):
    W1_X_transpose = np.transpose(W1_X) #make it 4*n
    W1_cov = np.empty([features,features])
    #print "Shape of class-1 transpose matrix: ", W1_X_transpose.shape
    test = []
    
    #print W1_X_transpose
    #print W1_X_transpose.shape
    W1_X_mean_subtract = np.subtract(W1_X_transpose, W1_mean)
    print ("class-1 after calculate subtract: ", W1_X_mean_subtract.shape)
    #print W1_X_mean_subtract
    elements = np.size(W1_X_mean_subtract,0)
    #print elements

    for i in range(np.size(W1_X_mean_subtract,1)):    #for features
        select1 = np.reshape(W1_X_mean_subtract[:,i],(elements,1))
        #print select1.shape
        select2 = select1.dot(np.transpose(select1))
        #print select2.shape
        test.append(np.array(select2))
        
    return mean_calulate2(W1_cov,test)

def find_cov_naive(W1_X, W1_mean,features):
    W1_X_transpose = np.transpose(W1_X) #make it 4*n
    W1_cov = np.empty([features,features])
    #print "Shape of class-1 transpose matrix: ", W1_X_transpose.shape
    test = []
    
    #print W1_X_transpose
    #print W1_X_transpose.shape
    W1_X_mean_subtract = np.subtract(W1_X_transpose, W1_mean)
    print ("class-1 after calculate subtract: ", W1_X_mean_subtract.shape)
    #print W1_X_mean_subtract
    elements = np.size(W1_X_mean_subtract,0)
    #print elements

    for i in range(np.size(W1_X_mean_subtract,1)):    #for features
        select1 = np.reshape(W1_X_mean_subtract[:,i],(elements,1))
        #print select1.shape
        select2 = select1.dot(np.transpose(select1))
        #print select2.shape
        test.append(np.array(select2))
    
    W1_cov = mean_calulate2(W1_cov,test)
    #I = np.identity(4)
    #print W1_cov
    #print I
    W1_cov = np.diag(np.diag(W1_cov))
    #print "xxxx:", W1_cov
    return W1_cov


def mean_calulate2(X,list):
    N = len(list) #0: total row
    #print N 
    summation = np.empty(X.shape)  
    #print summation.shape
    for i in range(N):
        summation = summation + list[i]
        #print summation
    cov = summation / (N-1) 
    cov = cov.astype(np.float64)  
    #fix for numpy.linalg.LinAlgError: SVD did not converge (nan)
    cov = np.nan_to_num(cov)        
    return np.array(cov)

def calculateClassifier(X,covInv1,covInv2,cov1,cov2,mean1,mean2):
    #print "classifier"
    A = (covInv2-covInv1)
    #print A.shape, X.shape, covInv1.shape, cov1.shape, mean1.shape  
    A = (np.transpose(X).dot(A)).dot(X)
    #print firstpart.shape 
    B = ((np.transpose(mean2).dot(covInv2))  -   (np.transpose(mean1).dot(covInv1)))
    #print B.shape
    B = -2 * (B.dot(X))
    
    if isnan(A) or isinf(A): A = 0
    if isnan(B) or isinf(B): B = 0  
     
    X2 = find_pseudodeterminant(cov2)
    #print X2
    variableLog2 = 0.0
    if (X2>0.0 and (isinstance(X2, complex) != True)):
        if isnan(np.log(X2)) or isinf(np.log(X2)): 
            variableLog2 = 0.0
        else: 
            variableLog2 = np.log(X2)
        
    X1 = find_pseudodeterminant(cov1)
    #print X1
    variableLog1 = 0.0
    if (X1>0.0 and (isinstance(X1, complex) != True)):
        if isnan(np.log(X1)) or isinf(np.log(X1)) or isinstance(X1, complex): 
            variableLog1 = 0.0
        else: 
            variableLog1 = np.log(X1)
        
    variableLog = variableLog2 - variableLog1
    
    C = variableLog + ((np.transpose(mean2).dot(covInv2)).dot(mean2)) - ((np.transpose(mean1).dot(covInv1)).dot(mean1))
    #print C   
    if isnan(A+B+C):
        return 0.0
    else:
        return int(round(A+B+C))

def find_accuracy(test_Y,test_pred):
    result = (test_Y == test_pred)
    trueDetection = (float)(np.sum(result))
    size = (int)(np.size(test_pred,0))
    #print trueDetection, size, trueDetection/size
    accuracy = (trueDetection/size)*100
    return accuracy


def find_pseudodeterminant(covarianceMatrix):
    #First compute the eigenvalues of your matrix    
    eig_values = np.linalg.eig(covarianceMatrix)
    #Then compute the product of the non-zero eigenvalues (this equals the pseudo-determinant value of the matrix)
    pseudo_determinent = np.product(eig_values[eig_values > 1e-12])
    #print pseudo_determinent
    return pseudo_determinent
