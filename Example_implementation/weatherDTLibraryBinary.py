# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:11:15 2019

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
from sklearn.preprocessing import LabelEncoder

#os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
################################################################################################################################################
#load data
#data = pd.read_csv("glass.data", sep=',', low_memory=False)
data = pd.read_csv('tennis.csv', names = ['day', 'outlook', 'temp', 'humidity', 'wind', 'play'])
data = data.drop('day',axis=1)
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
feature_cols = ['outlook', 'temp', 'humidity', 'wind']

################################################################################################################################################
def get_binary(dataset):
    
    features = dataset.shape[1]
    print (features)
    cols = ['outlook', 'temp', 'humidity', 'wind', 'play']
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

#data = get_binary(data) 
#removed head and index and convert to numpy array
data = data.iloc[1:, :].values 
np.random.shuffle(data) 
#print "Cleaned data:", data
print ("Cleaned data shape:", data.shape)

enc = LabelEncoder()

################################################################################################################################################
#data Training
features = np.size(data,1)-1 #column    [all columns except last one as it has predicted class]
samples = np.size(data,0)  #row
#print (features, samples)

for i in range(features):
    data[:,i] = enc.fit_transform(data[:,i])

print (data)
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
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = feature_cols, class_names= ['no','yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('glass2.bmp')
Image(graph.create_png())
