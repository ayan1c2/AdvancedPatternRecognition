# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:34:47 2019

@author: ayanca
"""
import numpy as np
import pandas as pd
from sklearn.impute  import SimpleImputer
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import confusion_matrix
#import os
from sklearn.preprocessing import Binarizer

#os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
################################################################################################################################################
#load data
#data = pd.read_csv("glass.data", sep=',', low_memory=False)
data = pd.read_csv("glass2.data", names = ['Index', 'RI', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron','class'])
data = data.drop('Index',axis=1)
#print ("Original data:", data)
print ("Original data shape:", data.shape)
#print data

#check if missing value (cleaning)
missing_values = ["n/a", "na", "--","?", " ","NA"]
data = data.replace(missing_values, np.nan)
feat_miss = data.columns[data.isnull().any()]
if feat_miss.size == 0:
    print ("Data is clean")
else:
    print ("Missing data shape before:", feat_miss.shape)
    imputer = SimpleImputer(copy=True, fill_value=None, missing_values=np.nan, strategy='mean', verbose=0)
    data[feat_miss] = imputer.fit_transform(data[feat_miss])
    feat_miss = data.columns[data.isnull().any()]
    print ("Missing data shape after:", feat_miss.shape)
    
#shuffle data
#data = data.sample(frac=1) #Returns a random sample of items from an axis of object.
print ("Shape after shuffling", data.shape)
feature_cols = ['Refractive index-1', 'Sodium-2', 'Magnesium-3', 'Aluminum-4', 'Silicon-5', 'Potassium-6', 'Calcium-7', 'Barium-8', 'Iron-9']
################################################################################################################################################
def get_binary(dataset):
    
    features = dataset.shape[1]
    print (features)
    cols = ['RI', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron','class']
    dataset = dataset.iloc[:,:]
    datacol = np.array(dataset.iloc[1:,-1])
    datacol = np.reshape(datacol,(datacol.shape[0],1))
    dataset =  dataset.iloc[1:,:features-1]              
    
    transformer = Binarizer().fit(dataset)
    dataset = transformer.transform(dataset) 
    dataset = np.reshape(dataset,(dataset.shape[0], dataset.shape[1]))
    
    dataset = np.concatenate((dataset, datacol), axis = 1)
    print (datacol.shape, dataset.shape)
    np.random.shuffle(dataset)
    dataset = pd.DataFrame(dataset, columns = cols)
    return dataset

data = get_binary(data) 
#removed head and index and convert to numpy array
data = data.iloc[:, :].values  
#print "Cleaned data:", data
print ("Cleaned data shape:", data.shape)

################################################################################################################################################
#data Training
features = np.size(data,1)-1 #column    [all columns except last one as it has predicted class]
samples = np.size(data,0)  #row
#print (features, samples)

#class finding
totalClass = data[:, features]
totalClass = (np.sort(np.unique(np.array(totalClass))))
totalClass = np.reshape(totalClass,[totalClass.size,1]) #convert to (*,1) array
print ("classes: ", totalClass)

################################################################################################################################################

X = data[:, :features]
y = data[:, features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

# Create Decision Tree classifer object
#clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("confusion matrix:",confusion_matrix(y_test, y_pred))
print("Accuracy with one fold:",(metrics.accuracy_score(y_test, y_pred)*100))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = feature_cols, class_names= ['1','2','3','5','6', '7'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('glass_1_fold.bmp')
Image(graph.create_png())

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
def calculate_accuracy(data_train,data_cv_test,i):
    X_train = data_train[:, :features]
    y_train = data_train[:, features]    
    
    X_test = data_cv_test[:, :features]
    y_test = data_cv_test[:, features]    
    
    #print (X_test, y_test)

    # Create Decision Tree classifer object
    #clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = DecisionTreeClassifier(criterion="entropy")

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("confusion matrix:",confusion_matrix(y_test, y_pred))
    print("Accuracy for DT with 1-fold:",(metrics.accuracy_score(y_test, y_pred)*100)) 
    
    #feature_cols = ['feature-1', 'feature-2', 'feature-3', 'feature-4', 'feature-5', 'feature-6', 'feature-7', 'feature-8', 'feature-9']
    
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = feature_cols, class_names= ['1','2','3','5', '6', '7'], )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('glass_5_fold.bmp')
    Image(graph.create_png())
    return (metrics.accuracy_score(y_test, y_pred)*100)

print ("For fold starts for DT")
accuracy = []
for i in range(fold-1):    
    
    training_idx = []    
    
    test_idx = splitArray[i]
    for j in range(len(splitArray)):
        if j !=i:
            training_idx.append(splitArray[i])
            
    training_idx = np.array(np.concatenate((training_idx), axis=0))
    print (training_idx, training_idx.shape)  

    data_train, data_cv_test = training_idx, test_idx
    print ("Train data Set: ", data_train.shape)    
    print ("CV Test data Set: ", data_cv_test.shape)  
  
    print ("For fold starts: ", (i+1))
    accuracyVal = calculate_accuracy(data_train,data_cv_test,(i+1))
    accuracy.append(accuracyVal)
    print ("For fold ends ")
   
################################################################################################################################################
print ("Average Cross Validation Accuracy for DT with 5-fold: ", sum(accuracy) / len(accuracy) )
################################################################################################################################################

