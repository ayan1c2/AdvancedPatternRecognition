# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:20:43 2019

@author: ayanca
"""

from random import random
import numpy as np
import pandas as pd

dataSample = 2000
d = 10
data = np.zeros([dataSample*4,d+1])

#first generate x3(root)
x3 =np.random.rand(4)
#x1, x5, x7 depwnds on x3; x1 = 0 | x3 = 0 and x1 = 0 | x3 = 1
x1 = np.random.rand(4, 2)
x5 = np.random.rand(4, 2)
x7 = np.random.rand(4, 2)
#x6, x9 depwnds on x1
x6 = np.random.rand(4, 2)
x9 = np.random.rand(4, 2)
#x2 depwnds on x5
x0 = np.random.rand(4, 2)
#x0 depwnds on x8
x2 = np.random.rand(4, 2)
#x4, x8 depwnds on x7
x4 = np.random.rand(4, 2)
x8 = np.random.rand(4, 2)

row = 0
for i in range (len(x3)):    
    for j in range (dataSample):
        if random() > x3[i]:
            data[row,3] = 1
            
        data[row,d] = i
        row += 1


row = 0
for i in range (len(x1)):    
    for j in range (dataSample):
        #x1
        if data[row,3] == 0 and random() > x1[i,0]:
            data[row,1] = 1
        if data[row,3] == 1 and random() > x1[i,1]:
            data[row,1] = 1
            
        #x5
        if data[row,3] == 0 and random() > x5[i,0]:
            data[row,5] = 1
        if data[row,3] == 1 and random() > x5[i,1]:
            data[row,5] = 1
            
        #x7
        if data[row,3] == 0 and random() > x7[i,0]:
            data[row,7] = 1
        if data[row,3] == 1 and random() > x7[i,1]:
            data[row,7] = 1            
        row += 1

row = 0
for i in range (len(x1)):    
    for j in range (dataSample):
        #x6
        if data[row,1] == 0 and random() > x6[i,0]:
            data[row,6] = 1
        if data[row,1] == 1 and random() > x6[i,1]:
            data[row,6] = 1
            
        #x9
        if data[row,1] == 0 and random() > x9[i,0]:
            data[row,9] = 1
        if data[row,1] == 1 and random() > x9[i,1]:
            data[row,9] = 1
        row += 1
        
row = 0
for i in range (len(x1)):    
    for j in range (dataSample):
        #x2
        if data[row,5] == 0 and random() > x2[i,0]:
            data[row,2] = 1
        if data[row,5] == 1 and random() > x2[i,1]:
            data[row,2] = 1
            
        row += 1


row = 0
for i in range (len(x1)):    
    for j in range (dataSample):
        #x4
        if data[row,7] == 0 and random() > x4[i,0]:
            data[row,4] = 1
        if data[row,7] == 1 and random() > x4[i,1]:
            data[row,4] = 1
            
        #x8
        if data[row,7] == 0 and random() > x8[i,0]:
            data[row,8] = 1
        if data[row,7] == 1 and random() > x8[i,1]:
            data[row,8] = 1
        row += 1


row = 0
for i in range (len(x1)):    
    for j in range (dataSample):
        #x0
        if data[row,8] == 0 and random() > x0[i,0]:
            data[row,0] = 1
        if data[row,8] == 1 and random() > x0[i,1]:
            data[row,0] = 1
            
        row += 1


import xlsxwriter
workbooko = xlsxwriter.Workbook('../data/generated_binary_data.xlsx')
worksheet = workbooko.add_worksheet() 
row = 0
for col, data in enumerate(data):
	worksheet.write_column(row, col, data)
workbooko.close()