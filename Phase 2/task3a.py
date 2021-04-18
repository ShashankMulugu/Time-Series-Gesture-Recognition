#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 19:08:09 2020

@author: mageshsridhar
"""

import ast
import glob
import os
import re
import collections

import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn import preprocessing

import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF


def similarity_func(u, v):
    return dot(u, v)/(norm(u)*norm(v))


def generateSimilarityMatrix(file):
    dists = pdist(file, similarity_func)
    DF_euclid = pd.DataFrame(squareform(dists), columns=file.index, index=file.index)
    if os.path.exists("./output/Similarity_matrix.csv"):  # files exit, delete them
        os.remove("./output/Similarity_matrix.csv")
    DF_euclid.to_csv("./output/Similarity_matrix.csv", sep=',')
    return DF_euclid


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


def score_computation(name, model, latent, principle_comp):
    pca_comp = []
    for i in range(principle_comp):
        scores = pd.Series(model.components_[i])
        sorted_scores = scores.abs().sort_values(ascending=False)

        final = []
        for sem, var in zip(latent[i], sorted_scores):
            fin = []
            fin = [sem, var]
            final.append(fin)

        pca_comp.append(final)

    tempdf = pd.DataFrame(pca_comp)
    tempdf.to_csv('./output/SVDGestureScoreOnSimilarityMatrix.csv', sep=",")

    labels = ['PRINCIPLE COMPONENT ' + str(x) + ':' for x in range(1, principle_comp + 1)]
    for label, pc in zip(labels, pca_comp):
        print(label)
        print(pc[:20])
        print('\n')


def display_topics(model, feature_names, no_top_words):
    res = []
    for topic_idx, topic in enumerate(model.components_):
        print("k = ", topic_idx , "  " , [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        res.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    return res


def performSVD(DF_euclid,principle_comp):
    svd_list = list(DF_euclid.columns.values)
    dfT = DF_euclid.T

    scaled_data = preprocessing.scale(DF_euclid)
    svd = TruncatedSVD(n_components=principle_comp)
    svd.fit(scaled_data)
    svd_data = svd.transform(scaled_data)

    print("\n")
    print("Latent Semantic Term - Weight Pairs: ")
    lat_semantics = display_topics(svd, svd_list, len(svd.components_[0]))
    print()
    score_computation('SVD', svd, lat_semantics,principle_comp)

    new_df = pd.DataFrame(svd_data)
    new_df = new_df.T
    new_df.columns = dfT.columns
    new_df = new_df.T
    new_df.to_csv('./output/SVDonSimilarityMatrix.csv', sep=",")
    svd_score = pd.DataFrame(svd.components_)
    svd_score.columns = svd_list
    svd_score.to_csv('./output/SVDScores.csv',header=True,sep=',')


if __name__ == "__main__":

    principle_comp = int(input("Enter the number of principle components : "))
    red_choice = input(
        "Enter the reduction technique of your choice \n 1->Dot Product 2->PCA 3->SVD 4->NMF 5->LDA 6->Edit distance 7->DTW : ")

    file = pd.DataFrame()
    if int(red_choice) == 1:
        if os.path.exists("./output/all_tf_gestures.csv"):
            file = pd.read_csv("./output/all_tf_gestures.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/all_tfidf_gestures.csv"):
            file = pd.read_csv("./output/all_tfidf_gestures.csv", index_col=0, header=0, sep=',')
        DF_euclid = generateSimilarityMatrixDP(file)
        if principle_comp > len(DF_euclid) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_euclid), " for top p principle components.")
            exit(1)
        performSVD(DF_euclid, principle_comp)

    if int(red_choice) == 2:
        if os.path.exists("./output/ReducedTF_PCA.csv"):
            file = pd.read_csv("./output/ReducedTF_PCA.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_PCA.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_PCA.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_euclid = generateSimilarityMatrix(file)
        if principle_comp > len(DF_euclid) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_euclid), " for top p principle components.")
            exit(1)
        performSVD(DF_euclid, principle_comp)

    if int(red_choice) == 3:
        if os.path.exists("./output/ReducedTF_SVD.csv"):
            file = pd.read_csv("./output/ReducedTF_SVD.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_SVD.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_SVD.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_euclid = generateSimilarityMatrix(file)
        if principle_comp > len(DF_euclid) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_euclid), " for top p principle components.")
            exit(1)
        performSVD(DF_euclid, principle_comp)

    if int(red_choice) == 4:
        if os.path.exists("./output/ReducedTF_NMF.csv"):
            file = pd.read_csv("./output/ReducedTF_NMF.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_NMF.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_NMF.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_euclid = generateSimilarityMatrix(file)
        if principle_comp > len(DF_euclid) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_euclid), " for top p principle components.")
            exit(1)
        performSVD(DF_euclid, principle_comp)

    if int(red_choice) == 5:
        if os.path.exists("./output/ReducedTF_LDA.csv"):
            file = pd.read_csv("./output/ReducedTF_LDA.csv", index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_LDA.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_LDA.csv", index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_euclid = generateSimilarityMatrix(file)
        if principle_comp > len(DF_euclid) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_euclid), " for top p principle components.")
            exit(1)
        performSVD(DF_euclid, principle_comp)

    if int(red_choice) == 6:
        DF_euclid = generateSimilarityMatrixED()
        if principle_comp > len(DF_euclid) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_euclid), " for top p principle components.")
            exit(1)
        performSVD(DF_euclid, principle_comp)

    if int(red_choice) == 7:
        DF_euclid = generateSimilarityMatrixDTW()
        if principle_comp > len(DF_euclid) or principle_comp < 1:
            print("Please enter a value between 1 and ",len(DF_euclid), " for top p principle components.")
            exit(1)
        performSVD(DF_euclid, principle_comp)