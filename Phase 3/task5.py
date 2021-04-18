# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 16:19:24 2020

@author: Chandu
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import collections
import copy
import os
import pickle 
import os.path
from os import path

def task5():
    
    def get_key(val): 
        for key, value in file_map.items(): 
             if val == value: 
                 return key 
         

################### Read output of task3 ... top t ####
    data = ''
    with open('./output/task3outputTopt.txt', "r") as f:
        data = f.read()
    d = data.split('\n')
    imagesNames = []
    reducerObjectpp = []
    for i in d:
        if i != '':
            imagesNames.append(i[i.find('<') + 1 : i.rfind(',')].replace(' ',''))
            reducerObjectpp.append(float(i[i.find(',') + 1 : i.rfind('>')].replace(' ','')))
    #print(imagesNames)
    #print(len(imagesNames))
    #print(reducerObjectpp)
##########################################


        
    ############### Get the similarity matrix of size t X t #####
    data = pd.read_csv(os.getcwd()+"/Similarity_matrix.csv",header=0,index_col=0)
    
    
    df = pd.DataFrame()
    for i in range(len(imagesNames)):
        row = str(imagesNames[i]) + ".wrd"
        li =[]
        for j in range(len(imagesNames)):
            col = str(imagesNames[j]) + ".wrd"
            value = data.loc[row,col]
            li.append(value)
            #print(li)
        d = pd.DataFrame([li])
        df = df.append(d)
    
    #df.rename(index = imagesNames, inplace = True)   
    names = []
    for x in imagesNames:
        names.append(str(x)+".wrd")
    df.columns = names
    df.index = names
    #######################################################
    
    ###################### Calculating the Trasition graph ###
    
    k = len(imagesNames)
    #s = int(input("Enter the number of seed nodes: "))
    filenames = list(df.columns)
    if path.exists("save_temp_out.p"):
        favorite_gestures = pickle.load( open( "save_temp_out.p", "rb" ) )
        if set(favorite_gestures) == set(filenames):
            filenames = favorite_gestures
    #else:
     #   print("NO")
    #print(filenames)
    
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
    A= nx.to_numpy_matrix(G)    #Adjaceny Matrix
    A = A/A.sum(axis=1)
    #print(A)
    c = 0.25
    ##########################################
    #print(file_map)
    ############# PPR ################
    ch = "n"
    iteration = 0
    #t_files = []
    #while ch == "n" or ch == "N":
    def subtask(filenames):        
       # iteration = iteration + 1
        numberOfRelavant = int(input("Number of relevant gestures : "))
        numberOfIrRelavant = int(input("Number of irrelevant gestures : "))
        print("Choose relavent and irrelavent gestures from the below list:")
        print(filenames)
        #if iteration == 1:
         #   print(filenames)
        #else:
         #   print(t_files)
        ##### relavant PPR
        seed_nodes = []
        for i in range(numberOfRelavant):
            print("Please enter the relavant gesture "+ str(i+1) +" :")
            seed_nodes.append((input()))
        
        v = np.zeros(N)
        for i in range(numberOfRelavant):
            v[get_key(str(seed_nodes[i])+".wrd")] = 1/numberOfRelavant
        v = np.array([v]).T
        u = copy.deepcopy(v)
        temp = copy.deepcopy(u)
        while True:
            u = ((1-c)*np.matmul(A,u)) + (c*v)
            if np.array_equal(temp,u):
                break
            temp = u
        u = u.flatten().tolist()[0]
        
        ### irrelavant PPR
        seed_nodes_2 = []
        for i in range(numberOfIrRelavant):
            print("Please enter the irrelavant gesture "+ str(i+1) +" :")
            seed_nodes_2.append((input()))
        
        v2 = np.zeros(N)
        for i in range(numberOfIrRelavant):
            v2[get_key(str(seed_nodes_2[i])+".wrd")] = 1/numberOfIrRelavant
        v2 = np.array([v2]).T
        u2 = copy.deepcopy(v2)
        temp2 = copy.deepcopy(u2)
        while True:
            u2 = ((1-c)*np.matmul(A,u2)) + (c*v2)
            if np.array_equal(temp2,u2):
                break
            temp2 = u2
        u2 = u2.flatten().tolist()[0]
        
        #print(u)
        #print(u2)
        finalResult = {}
        for i in range(len(imagesNames)):
            finalResult[file_map[i]] = u[i] - u2[i]
        
        #print(finalResult)
        sorted_dict = dict( sorted(finalResult.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True))
        #finalResult = list(dict(sortList).keys())
        #print(sorted_dict)
        
        #for key, value in sorted_dict.items():
         #   print("< " + str(key) + ", " + str(value) + " >")
        
        finalkeys = list(sorted_dict.keys())
        print("Revised List of results after PPR feedback:")
        print(finalkeys)
        t_files = finalkeys
        return t_files
        #ch = input("Are you satisfied with the output? type Y for exit N for running again ")
    
    #while(1):
    out = subtask(filenames)
    pickle.dump( out, open( "save_temp_out.p", "wb" ) )
    #filenames = out
     

task5()