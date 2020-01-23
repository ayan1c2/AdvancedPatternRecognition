"""
Dependent Binary features and Dependet tree
"""
import random
import numpy as np
import math
from sklearn.metrics import confusion_matrix

percentile = 33
df = np.loadtxt("glass.txt").astype(dtype=np.float32)
dataframeoriginal = df[0:214,1:11] 
dataframeoriginal = np.delete(dataframeoriginal, 7, 1)
dataframeoriginal = np.delete(dataframeoriginal, 7, 1)

f = dataframeoriginal.shape[1]-1
dataframe = np.zeros([1,dataframeoriginal.shape[1]])
for kk in range(dataframeoriginal.shape[0]):
    if dataframeoriginal[kk,f] == 1 or dataframeoriginal[kk,f] == 2 or dataframeoriginal[kk,f] == 3 or dataframeoriginal[kk,f] == 7:
        dataframe = np.append(dataframe, dataframeoriginal[kk,:].reshape((1, dataframe.shape[1])), axis=0)
        
dataframe = dataframe[1:len(dataframe),:]
np.random.shuffle(dataframe)

dataset = np.zeros((len(dataframe), len(dataframe[0])))

m = -1
for i in range(len(dataframe[0])-1):
    for k in range(len(dataframe)):
        if dataframe[k,i] <= np.percentile(dataframe[:,i], 33):
            dataset[k,i] = 0
        elif np.percentile(dataframe[:,i], 33) < dataframe[k,i] <= np.percentile(dataframe[:,i], 50):
            dataset[k,i] = 1
        elif np.percentile(dataframe[:,i], 50) < dataframe[k,i] <= np.percentile(dataframe[:,i], 77):
            dataset[k,i] = 2
        else:
            dataset[k,i] = 3
                
dataset[:,len(dataframe[0])-1] = dataframe[:,len(dataframe[0])-1]

NoofFeatures = dataset.shape[1]-1
classes = list(set(dataset[:,NoofFeatures]))
classes.sort() 
NoofFolds = 5

samples = 200
d = 7

#--------------------------------------functions----------------------------------------

def fit(class1features):
    #recieve complete training matrix
    weights = np.zeros([class1features.shape[1],class1features.shape[1]])
    d = 1
    for i in range(class1features.shape[1]-1):
        for j in range(d,class1features.shape[1]):
            feature1 = class1features[:,i]
            feature2 = class1features[:,j]
            #print(feature1)
            
            feature1Unique = list(set(feature1))
            feature2Unique = list(set(feature2))
            #print(feature1Unique)
            weight = 0
            for mm in feature1Unique:
                for kk in feature2Unique:
                    count = 0
                    countmm = 0
                    countkk = 0
                    for row in range(class1features.shape[0]):
                        if feature1[row] == mm and feature2[row] == kk:
                            count += 1
                            
                        if feature1[row] == mm:
                            countmm += 1
                            
                        if feature2[row] == kk:
                            countkk += 1
                                            
                    Ptot = count/class1features.shape[0]
                    Pmm = countmm/class1features.shape[0]
                    Pkk = countkk/class1features.shape[0]
                    if Ptot == 0 or Pmm == 0 or Pkk == 0:
                        weight += 0
                    else:
                        weight += (Ptot) * math.log(Ptot/(Pmm*Pkk))
                            
            weights[i,j] = weight
        d += 1
        
    return weights

def ProbCalculator(features, root, child, sample):
    root = int(root)
    child = int(child)
    if child == -1:
        top = 0
        for gg in range(features.shape[0]):
            if features[gg,root] == sample[root]:
                top += 1
        probability = top / features.shape[0]
    else:
        base = 0
        top = 0
        for gg in range(features.shape[0]):
            if features[gg,root] == sample[root]:
                base += 1
                if features[gg,child] ==sample[child]:
                    top += 1
        if base == 0:
            return 0
        else:
            probability = top/base           
        
    return probability
    
def getProbability(features, connections0, root, P, sample):
    sigma = []
    #print('root',root)
    pp = 0
    for i in range(connections0.shape[0]):
        if connections0[pp, 0] == root:
            sigma = np.append(sigma, connections0[pp, 1])
            connections0 = np.delete(connections0, pp, axis=0)
            pp -= 1
        
        elif connections0[pp, 1] == root:
            sigma = np.append(sigma, connections0[pp, 0])
            connections0 = np.delete(connections0, pp, axis=0)
            pp -= 1        
        pp += 1
        
    #print(sigma)
    #print('kæ')
    if any(sigma):
        for jj in sigma:
            P = P * ProbCalculator(features, root, jj, sample)
            
        for newroot in sigma:
            getProbability(features, connections0, newroot, P, sample)
                
    else:
        return P
    return P
            
def getconnections(trainingfeatues,weights):
    connections1 = []
    count = 0
    for i in range(trainingfeatues.shape[1]*trainingfeatues.shape[1]):
        link = np.unravel_index(np.argmax(weights, axis=None), weights.shape)
#        print(link, weights[link])
        a, b = link
        weights[link] = 0
        
        if i == 0:
            connections1.append(a)
            connections1.append(b)
            
        else:
#            if a not in list(set(connections1)) or b not in list(set(connections1)):
#                connections1.append(a)
#                connections1.append(b)
#                count += 1
                
#            if a in list(set(connections1)) or b in list(set(connections1)):                       
            if a in list(set(connections1)) and b in list(set(connections1)):
                continue
            else:
                connections1.append(a)
                connections1.append(b)
                count += 1
        if count == 8:
            continue
    
    root = random.choice(list(set(connections1)))
    connections1 = np.asarray(connections1).reshape((count+1, 2))
    return root, connections1
    
def predict(trainingfeatues, trainingvariables, weights, sample, classes, root, connections):
    firstclass = 0
    secondclass = 1
    
    for test in range(len(classes)-1):
        
        featureC1 = np.zeros([1,trainingfeatues.shape[1]])
        for kk in range(trainingfeatues.shape[0]):
            if trainingvariables[kk] == classes[firstclass]:
                featureC1 = np.append(featureC1, trainingfeatues[kk,:].reshape((1, features.shape[1])), axis=0)
            
        featureC2 = np.zeros([1,trainingfeatues.shape[1]])
        for kk in range(trainingfeatues.shape[0]):
            if trainingvariables[kk] == classes[secondclass]:
                featureC2 = np.append(featureC2, trainingfeatues[kk,:].reshape((1, features.shape[1])), axis=0)           
        
        featureC1 = featureC1[1:len(featureC1),:]
        featureC2 = featureC2[1:len(featureC2),:]
        
        RootProbabilityC1 = ProbCalculator(featureC1, root, -1, sample)
        RootProbabilityC2 = ProbCalculator(featureC2, root, -1, sample)

        class1prob = getProbability(featureC1, connections, root, RootProbabilityC1, sample)
        class2prob = getProbability(featureC2, connections, root, RootProbabilityC2, sample)        
        
        #print(clasifierValue)
        if class1prob > class2prob:
            firstclass = firstclass
        else:
            firstclass = secondclass
            
        secondclass = secondclass + 1
            
    #print(firstclass)    
    return firstclass
        
#test
def Tester(trainingfeatues, trainingvariables, sample, lables, weights, classes, root, connections):
    error = 0
    predicted = np.zeros([1,1])
    for i in range(sample.shape[0]):
        predicted = np.append(predicted, classes[predict(trainingfeatues, trainingvariables, weights, sample[i,:], classes, root, connections)].reshape((1, 1)), axis=0)
        if lables[i] != classes[predict(trainingfeatues, trainingvariables, weights, sample[i,:], classes, root, connections)]:
            error += 1
    
    Accuracy = 1.0 - 1.0 * error / sample.shape[0]
    predicted = predicted[1:len(predicted),:]
    #print(predicted)
    return Accuracy*100, predicted


#--------------------------------------main program ------------------------------------------

predictedy = np.zeros([1,1])
Actual = np.zeros([1,1])
start = 0
AccAcumlt = np.zeros([NoofFolds,2])
for folds in range(NoofFolds):
    
    features = dataset[:,0:d]
    
    AllFeatureIndex = np.asarray(list(range(0, features.shape[0])))
    TestingIndex = np.asarray(list(range(start, start+features.shape[0]//NoofFolds)))
    TrainingIndex = np.asarray(list(set(AllFeatureIndex)-set(TestingIndex)))
    
    trainingfeatues = np.zeros([1,features.shape[1]])
    trainingvariables = np.zeros([1,1])
    for kk in range(features.shape[0]):
        if kk in TrainingIndex:
            trainingfeatues = np.append(trainingfeatues, features[kk,:].reshape((1, features.shape[1])), axis=0)
            trainingvariables = np.append(trainingvariables, dataset[kk,d].reshape((1, 1)), axis=0)
            
    testingfeatures = np.zeros([1,features.shape[1]])
    testingvariables = np.zeros([1,1])
    for kk in range(features.shape[0]):
        if kk in TestingIndex:
            testingfeatures = np.append(testingfeatures, features[kk,:].reshape((1, features.shape[1])), axis=0)
            testingvariables = np.append(testingvariables, dataset[kk,d].reshape((1, 1)), axis=0)
            
    trainingfeatues = trainingfeatues[1:len(trainingfeatues),:]
    testingfeatures = testingfeatures[1:len(testingfeatures),:]
    
    trainingvariables = trainingvariables[1:len(trainingvariables),:]
    testingvariables = testingvariables[1:len(testingvariables),:]
    
    weights = fit(trainingfeatues)
#    print(weights)
    
    root, connections = getconnections(trainingfeatues,weights)
#    print(root, connections)
        
    AccAcumlt[folds,0],ptraining = Tester(trainingfeatues, trainingvariables, trainingfeatues, trainingvariables, weights, classes, root, connections)
    AccAcumlt[folds,1],ptesting = Tester(trainingfeatues, trainingvariables, testingfeatures, testingvariables, weights, classes, root, connections)
    
    for jj in range(len(ptesting)):
        predictedy = np.append(predictedy, ptesting[jj,0].reshape((1, 1)), axis=0)
        Actual = np.append(Actual, testingvariables[jj,0].reshape((1, 1)), axis=0)
    
    start =  start + features.shape[0]//NoofFolds

#AverageTrgAcc = sum(AccAcumlt[:,0])/NoofFolds
#AverageTsgAcc = sum(AccAcumlt[:,1])/NoofFolds

print('Average Training Accuracy = ',sum(AccAcumlt[:,0])/NoofFolds)
print('Average Testing Accuracy = ',sum(AccAcumlt[:,1])/NoofFolds)
predictedy = predictedy[1:len(predictedy),:]
Actual = Actual[1:len(Actual),:]
print(confusion_matrix(Actual, predictedy))


