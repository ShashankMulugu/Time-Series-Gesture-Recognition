#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:51:08 2020

@author: mageshsridhar
"""

import pandas as pd
import numpy as np
from math import sqrt
import glob
import os
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn import preprocessing
from numpy import dot
from numpy.linalg import norm


def cosineean_dist(X1,X2):
    dist = []
    for i in range(len(X1)):
        dist.append((X1[i]-X2[i])**2)
    return sqrt(sum(dist))

def kmeans(data,k):
    iterations = 10
    flag = 0
    centers = data.sample(n=k)
    for n in range(iterations):
        cluster_labels = []
        clusters = {}
        for i in range(len(data)):
            X = np.array(data.iloc[i])
            cluster_dist = []
            for j in range(len(centers)):
                C = np.array(centers.iloc[j])
                cluster_dist.append(cosineean_dist(X,C))
            cluster_labels.append(cluster_dist.index(min(cluster_dist)))
        for l in range(k):
            clusters[l] = []
        for i in range(k):
            for j in range(len(cluster_labels)):
                if i==cluster_labels[j]:
                    clusters[i].append(j+1)
        for i in range(len(centers)):
            prev_center = list(centers.iloc[i])
            new_center = np.zeros(len(data))
            for m in clusters[i]:
                new_center = new_center + data.iloc[m-1]
            new_center = new_center/len(clusters[i])
            if prev_center == list(new_center):
                flag = 1
            else:
                flag = 0
            centers.iloc[i] = new_center
        if flag==1:
            break
    return clusters




def similarity_func(u, v):
    return dot(u, v)/(norm(u)*norm(v))


def generateSimilarityMatrix(file):
    dists = pdist(file, similarity_func)
    DF_cosine = pd.DataFrame(squareform(dists), columns=file.index, index=file.index)
    if os.path.exists("./output/Similarity_matrix.csv"):  # files exit, delete them
        os.remove("./output/Similarity_matrix.csv")
    DF_cosine.to_csv("./output/Similarity_matrix.csv", sep=',')
    return DF_cosine


def generateSimilarityMatrixED():
    filCount = 0
    fileList = []
    for fil in glob.glob("./output/task2/edit_dist_*.txt"):
        filCount+=1
        filName = fil.replace("./output/task2/edit_dist_", "")
        filName = filName.replace(".txt", ".wrd")
        fileList.append(filName)
    dat = pd.DataFrame([[0] * filCount] * filCount, columns=fileList, index=fileList)
    for fil in glob.glob("./output/task2/edit_dist_*.txt"):
        edFile = open(fil, "r")
        edFileData = edFile.read()
        edFileData = edFileData[28:]
        edFileData = eval(edFileData[:-2])
        filName = fil.replace("./output/task2/edit_dist_", "")
        filName = filName.replace(".txt", ".wrd")

        for i in edFileData:
            for j in edFileData[i]:
                dat.loc[filName,str(j)+".wrd"] = i
    if os.path.exists("./output/Similarity_matrix.csv"):  # files exit, delete them
        os.remove("./output/Similarity_matrix.csv")
    dat.to_csv("./output/Similarity_matrix.csv", sep=',')
    return dat


def generateSimilarityMatrixDTW():
    filCount = 0
    fileList = []
    for fil in glob.glob("./output/task2/dtw_dist_*.txt"):
        filCount += 1
        filName = fil.replace("./output/task2/dtw_dist_", "")
        filName = filName.replace(".txt", ".wrd")
        fileList.append(filName)
    dat = pd.DataFrame([[0] * filCount] * filCount, columns=fileList, index=fileList)
    for fil in glob.glob("./output/task2/dtw_dist_*.txt"):
        edFile = open(fil, "r")
        edFileData = edFile.read()
        edFileData = edFileData[28:]
        edFileData = eval(edFileData[:-2])
        filName = fil.replace("./output/task2/dtw_dist_", "")
        filName = filName.replace(".txt", ".wrd")

        for i in edFileData:
            for j in edFileData[i]:
                dat.loc[filName, str(j) + ".wrd"] = i
    if os.path.exists("./output/Similarity_matrix.csv"):  # files exit, delete them
        os.remove("./output/Similarity_matrix.csv")
    dat.to_csv("./output/Similarity_matrix.csv", sep=',')
    return dat


def generateSimilarityMatrixDP(file):
    dat = pd.DataFrame([[0] * len(file)] * len(file), columns=file.index, index=file.index)
    for i in range(len(file)):
        for j in range(len(file)):
            dat.iloc[i,j] = np.dot(file.iloc[i],file.iloc[j])
    if os.path.exists("./output/Similarity_matrix.csv"):  # files exit, delete them
        os.remove("./output/Similarity_matrix.csv")
    dat.to_csv("./output/Similarity_matrix.csv", sep=',')
    return dat


if __name__ == "__main__":
    
    principle_comp = len(pd.read_csv("./output/SVDScores.csv",header=0,index_col=0,sep=','))
    red_choice = input("Enter the reduction technique of your choice \n 1->Dot Product 2->PCA 3->SVD 4->NMF 5->LDA 6->Edit distance 7->DTW : ")

    file = pd.DataFrame()
    if int(red_choice) == 1:
        if os.path.exists("./output/all_tf_gestures.csv"):
            file = pd.read_csv("./output/all_tf_gestures.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/all_tfidf_gestures.csv"):
            file = pd.read_csv("./output/all_tfidf_gestures.csv", index_col=0, header=0, sep=',')
        DF_cosine = generateSimilarityMatrixDP(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 2:
        if os.path.exists("./output/ReducedTF_PCA.csv"):
            file = pd.read_csv("./output/ReducedTF_PCA.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_PCA.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_PCA.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 3:
        if os.path.exists("./output/ReducedTF_SVD.csv"):
            file = pd.read_csv("./output/ReducedTF_SVD.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_SVD.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_SVD.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 4:
        if os.path.exists("./output/ReducedTF_NMF.csv"):
            file = pd.read_csv("./output/ReducedTF_NMF.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_NMF.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_NMF.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 5:
        if os.path.exists("./output/ReducedTF_LDA.csv"):
            file = pd.read_csv("./output/ReducedTF_LDA.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_LDA.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_LDA.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 6:
        DF_cosine = generateSimilarityMatrixED()
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 7:
        DF_cosine = generateSimilarityMatrixDTW()
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_cosine), " for top p principle components.")
            exit(1)
    
    
    filenames = DF_cosine.columns.values
    filemap = {}
    for i in range(len(DF_cosine)):
        filemap[i] = filenames[i]
    clusters = kmeans(DF_cosine,principle_comp)
    for i in range(len(clusters)):
        print()
        print("Cluster "+str(i+1)+":")
        for j in clusters[i]:
            print(filemap[j-1])

#NMF SCORE
#Task2 file generation for all options
#task2 increase NMF convergence limit

