# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:45:42 2019

@author: ayanca
"""

import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import confusion_matrix
import pydot

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
dataset = pd.read_csv("glass_artificial.csv", sep=',', low_memory=False)
dataset = dataset.iloc[:, 1:].values

dataset = pd.DataFrame(dataset, columns = cols)
dataset = get_binary(dataset)  
#print (dataset)
target_col_name = "class"
classValue = np.unique(dataset[target_col_name])
#####################################################################################################################################################################
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name="class"):
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def Dtree(data,originaldata,features,target_attribute_name="class",parent_node_class = None):     
    #Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#
    
    #If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    #If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    #the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    #the mode target feature value is stored in the parent_node_class variable.
    
    elif len(features) ==0:
        return parent_node_class
    
    #If none of the above holds true, grow the tree!
    
    else:
        #Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        #gain in the first run
        tree = {best_feature:{}}
        
        
        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = Dtree(sub_data,sub_data,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)    
 
def predict(query,tree,default = 1):    
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
        
def train_test_split(dataset):
    training_data = dataset.iloc[:select_data].reset_index(drop=True)#We drop the index respectively relabel the index
    #starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_data = dataset.iloc[select_data:].reset_index(drop=True)
    return training_data,testing_data

def test(data,tree):
    #Create new query instances by simply removing the target feature column from the original dataset and 
    #convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print ('The confusion matrix - ', confusion_matrix(predicted["predicted"], data[target_col_name]))
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data[target_col_name])/len(data))*100,'%')
    

#data Training
features = np.size(dataset,1)-1 #column    [all columns except last one as it has predicted class]
samples = np.size(dataset,0)  #row
select_data = (round)(samples*.8)

training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1] 

tree = Dtree(training_data,training_data,training_data.columns[:-1])
pprint(tree)
test(testing_data,tree)

#############################################plotting#############################################################
def draw(parent_name, child_name):
    edge = pydot.Edge(str(parent_name), str(child_name))   
    #print (parent_name, child_name)
    if child_name == "play":
        print("")
    else:
        graph.add_edge(edge)   
    

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, v)

graph = pydot.Dot(graph_type='graph')
visit(tree)
#graph.write_png('example2_graph.png')

#app_json = json.dumps(tree)
#print(app_json)
#############################################done#############################################################