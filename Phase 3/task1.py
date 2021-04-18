#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:35:41 2020

@author: mageshsridhar
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import copy
import os
import json

def get_key(val): 
    for key, value in file_map.items(): 
         if val == value: 
             return key 
df = pd.read_csv(os.getcwd()+"/Similarity_matrix.csv",header=0,index_col=0)
k = int(input("Enter the value for k: "))
s = int(input("Enter the number of seed nodes: "))
filenames = list(df.columns)
N = len(filenames)
file_map = {}
for i in range(len(filenames)):
    file_map[i] = filenames[i]
top_k_values = dict()
top_k_files = dict()
for i in filenames:
    temp_dict = {}
    temp_k=k
    for j in filenames:
        temp_dict[j] = df.loc[i][j]
    sorted_temp_dict = sorted(temp_dict.items(), key=lambda kv: kv[1],reverse=True)
    sorted_temp_dict = collections.OrderedDict(sorted_temp_dict)
    for n in sorted_temp_dict.keys():
        if temp_k>0:
            if i in top_k_values:
                top_k_values[i].append(sorted_temp_dict[n])
                top_k_files[i].append(n)
            else:
                top_k_values[i]=[sorted_temp_dict[n]]
                top_k_files[i]=[n]
                
            temp_k-=1
G = nx.DiGraph()
edge_list = []
for i in top_k_values.keys():
    for j in range(k):
        edge_list.append((i,top_k_files[i][j],top_k_values[i][j]))
G.add_weighted_edges_from(edge_list)
#nx.draw_networkx(G, pos = nx.spring_layout(G,k=1,iterations=2000),nxarrows=True,with_labels=True,node_size=200,linewidth=0.5,width=0.2,font_size=7,font_color="r",arrow_size=3)        

json.dump(top_k_values, open("./top_k_values.json", 'w'))    
json.dump(top_k_files, open("./top_k_files.json", 'w'))    

A= nx.to_numpy_matrix(G)    #Adjaceny Matrix
A = A/A.sum(axis=1)
#print(A)
c = 0.15                    #Restart Probability
print("Enter the file names from the following without extension that should be seed nodes (Eg.262)")
for i in filenames:
    print(i,end="\t")

seed_nodes = []
for i in range(s):
    seed_nodes.append(int(input()))

v = np.zeros(N)
for i in range(s):
    v[get_key(str(seed_nodes[i])+".wrd")] = 1/s
v = np.array([v]).T
u = copy.deepcopy(v)
temp = copy.deepcopy(u)
while True:
    u = ((1-c)*np.matmul(A,u)) + (c*v)
    if np.array_equal(temp,u, equal_nan=True):
        break
    temp = u
u = u.flatten().tolist()[0]
m = int(input("Enter the number of most dominant gestures you want to visualize (m): "))
dominant_files = []
for i in range(m):
    dominant_files.append(file_map[u.index(max(u))])
    u[u.index(max(u))] = -1

column_names = []
for i in range(20):
    column_names.append("Series "+str(i+1))


print(dominant_files)
for file in dominant_files:
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    w_file_df = pd.read_csv(os.getcwd()+"/3_class_gesture_data/W/"+file.replace(".wrd",".csv"),header=None).transpose()
    w_file_df.columns = column_names
    w_file_df.plot(title = file+" W Component",ax = ax1)
    x_file_df = pd.read_csv(os.getcwd()+"/3_class_gesture_data/X/"+file.replace(".wrd",".csv"),header=None).transpose()
    x_file_df.columns = column_names
    x_file_df.plot(title = file+" X Component",ax = ax2)
    y_file_df = pd.read_csv(os.getcwd()+"/3_class_gesture_data/Y/"+file.replace(".wrd",".csv"),header=None).transpose()
    y_file_df.columns = column_names
    y_file_df.plot(title = file+" Y Component",ax = ax3)
    z_file_df = pd.read_csv(os.getcwd()+"/3_class_gesture_data/Z/"+file.replace(".wrd",".csv"),header=None).transpose()
    z_file_df.columns = column_names
    z_file_df.plot(title = file+" Z Component",ax = ax4)
    plt.show()
    

        
        
            
            
