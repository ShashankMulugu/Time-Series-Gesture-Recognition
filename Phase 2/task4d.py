from os import listdir, mkdir
from os.path import isfile, join
from math import sqrt
from scipy.spatial.distance import cdist, pdist, squareform
from collections import defaultdict
from os import path
import scipy
import numpy
import glob
import os
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd


def float_formatter(x): return "%.3f" % x


np.set_printoptions(formatter={'float_kind': float_formatter})


def similarity_func(u, v):
    return dot(u, v)/(norm(u)*norm(v))


def generateSimilarityMatrix(file):
    dists = pdist(file, similarity_func)
    DF_cosine = pd.DataFrame(squareform(
        dists), columns=file.index, index=file.index)
    if os.path.exists("./output/Similarity_matrix.csv"):  # files exit, delete them
        os.remove("./output/Similarity_matrix.csv")
    DF_cosine.to_csv("./output/Similarity_matrix.csv", sep=',')
    return DF_cosine


def generateSimilarityMatrixED():
    filCount = 0
    fileList = []
    for fil in glob.glob("./output/task2/edit_dist_*.txt"):
        filCount += 1
        filName = fil.replace("./output/task2/edit_dist_", "")
        filName = filName.replace(".txt", ".wrd")
        fileList.append(filName)
    dat = pd.DataFrame([[0] * filCount] * filCount,
                       columns=fileList, index=fileList)
    for fil in glob.glob("./output/task2/edit_dist_*.txt"):
        edFile = open(fil, "r")
        edFileData = edFile.read()
        edFileData = edFileData[28:]
        edFileData = eval(edFileData[:-2])
        filName = fil.replace("./output/task2/edit_dist_", "")
        filName = filName.replace(".txt", ".wrd")

        for i in edFileData:
            for j in edFileData[i]:
                dat.loc[filName, str(j)+".wrd"] = i
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
    dat = pd.DataFrame([[0] * filCount] * filCount,
                       columns=fileList, index=fileList)
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
    dat = pd.DataFrame([[0] * len(file)] * len(file),
                       columns=file.index, index=file.index)
    for i in range(len(file)):
        for j in range(len(file)):
            dat.iloc[i, j] = np.dot(file.iloc[i], file.iloc[j])
    if os.path.exists("./output/Similarity_matrix.csv"):  # files exit, delete them
        os.remove("./output/Similarity_matrix.csv")
    dat.to_csv("./output/Similarity_matrix.csv", sep=',')
    return dat


def kmeans(X, k=3, max_iterations=50):
    if isinstance(X, pd.DataFrame):
        X = X.values
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx, :]
    P = np.argmin(cdist(X, centroids, 'euclidean'), axis=1)
    for _ in range(max_iterations):
        centroids = np.vstack([X[P == i, :].mean(axis=0) for i in range(k)])
        tmp = np.argmin(cdist(X, centroids, 'euclidean'), axis=1)
        if np.array_equal(P, tmp):
            break
        P = tmp
    return P


def SpectralClustering(sim_matrix, query_p):
    vectorizer = np.vectorize(
        lambda x: 1 if x < sim_matrix.stack().mean() else 0)
    W = np.vectorize(vectorizer)(sim_matrix)  # Adjacency matrix
    D = np.diag(np.sum(np.array(W), axis=1))  # Degree matrix
    L = D - W  # Laplacian matrix

    e, v = np.linalg.eig(L)  # eigenvalues, eigenvectors
    i = np.where(e == np.partition(e, 1)[1])
    print(i)
    U = np.array(v[:, i])
    U_df = pd.DataFrame(data=U.reshape(-1, 1))
    clus_outputs = kmeans(U_df, query_p)
    print("Spectral Cluster Lables for all files based on p = "+str(query_p))
    print(clus_outputs)
    return clus_outputs


if __name__ == "__main__":
    data_dir = input("Enter data directory name: ")
    mypath = data_dir + "\\W"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith('.csv')]
    file_names = []
    for i in range(len(onlyfiles)):
        file_names.append(str(onlyfiles[i].replace(".csv", "")))
    file_dict = {index-1: x for index, x in enumerate(file_names, start=1)}

    principle_comp = len(pd.read_csv(
        "./output/SVDScores.csv", header=0, index_col=0, sep=','))
    red_choice = input(
        "Enter the reduction technique of your choice \n 1->Dot Product 2->PCA 3->SVD 4->NMF 5->LDA 6->Edit distance 7->DTW : ")

    file = pd.DataFrame()
    if int(red_choice) == 1:
        if os.path.exists("./output/all_tf_gestures.csv"):
            file = pd.read_csv("./output/all_tf_gestures.csv",
                               index_col=0, header=0, sep=',')
        elif os.path.exists("./output/all_tfidf_gestures.csv"):
            file = pd.read_csv("./output/all_tfidf_gestures.csv",
                               index_col=0, header=0, sep=',')
        DF_cosine = generateSimilarityMatrixDP(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ", len(
                DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 2:
        if os.path.exists("./output/ReducedTF_PCA.csv"):
            file = pd.read_csv("./output/ReducedTF_PCA.csv",
                               index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_PCA.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_PCA.csv",
                               index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ", len(
                DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 3:
        if os.path.exists("./output/ReducedTF_SVD.csv"):
            file = pd.read_csv("./output/ReducedTF_SVD.csv",
                               index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_SVD.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_SVD.csv",
                               index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ", len(
                DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 4:
        if os.path.exists("./output/ReducedTF_NMF.csv"):
            file = pd.read_csv("./output/ReducedTF_NMF.csv",
                               index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_NMF.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_NMF.csv",
                               index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ", len(
                DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 5:
        if os.path.exists("./output/ReducedTF_LDA.csv"):
            file = pd.read_csv("./output/ReducedTF_LDA.csv",
                               index_col=0, header=0, sep=',')
        elif os.path.exists("./output/ReducedTFIDF_LDA.csv"):
            file = pd.read_csv("./output/ReducedTFIDF_LDA.csv",
                               index_col=0, header=0, sep=',')
        else:
            print('Please execute task 1 for this option')
            exit(1)
        DF_cosine = generateSimilarityMatrix(file)
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ", len(
                DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 6:
        DF_cosine = generateSimilarityMatrixED()
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ", len(
                DF_cosine), " for top p principle components.")
            exit(1)

    if int(red_choice) == 7:
        DF_cosine = generateSimilarityMatrixDTW()
        if principle_comp > len(DF_cosine) or principle_comp < 1:
            print("Please enter a value between 1 and ", len(
                DF_cosine), " for top p principle components.")
            exit(1)

    clust_op = SpectralClustering(DF_cosine, principle_comp)
    clust_dict = defaultdict(list)

    for i in range(len(file_names)):
        clust_dict[clust_op[i]].append(file_names[i])

    print("\nThe input files are in the following clusters: \n")
    print(dict(clust_dict))
