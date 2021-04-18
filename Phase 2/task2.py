from os import listdir, mkdir
from os.path import isfile, join
from collections import defaultdict
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cosine
from scipy.spatial.distance import cdist
from math import isinf
import numpy as np
from pathlib import Path
import pandas as pd
import time


def useDimReductionAlgs(q_file, vector_model, d_metric):
    dim_red_df = pd.read_csv("output\\Reduced"+vector_model +
                             "_"+d_metric+".csv", index_col=0)
    dfT = dim_red_df.T
    sim = []
    for column in dfT:
        if column != q_file:
            sim.append(
                [column, 1 - cosine(pd.to_numeric(dfT[q_file]), pd.to_numeric(dfT[column]))])
    results = sorted(sim,  key=lambda x: x[1], reverse=True)[:10]

    print("\n\n Most Similar 10 gesture files for = ", q_file)
    print("\nUSING COSINE SIMILARITY:")
    for row in results:
        print(row)
    print()


def findDotProductSimilarity(q_file, vector_model_query):
    dpTopDict = defaultdict(list)
    for i in range(len(file_names)):
        if q_file != file_names[i]:
            q_file_dpList = genDotProductList(q_file, vector_model_query)
            curr_file_dpList = genDotProductList(
                file_names[i], vector_model_query)
            if len(q_file_dpList) <= len(curr_file_dpList):
                dpTopDict[np.dot(q_file_dpList, curr_file_dpList)].append(
                    file_names[i])
            else:
                dpTopDict[editdistance(curr_file_dpList, q_file_dpList)].append(
                    file_names[i])
    dists = []
    for k in dpTopDict.keys():
        dists.append(k)
    dists.sort(reverse=True)
    topTenFilesList = []
    for i in range(len(dists)):
        topTenFilesList.extend(dpTopDict[dists[i]])
    return dpTopDict, topTenFilesList[:10]


def findEditDistance(q_file):
    edTopDict = defaultdict(list)
    for i in range(len(file_names)):
        if q_file != file_names[i]:
            q_file_EdList = genEditDistanceList(q_file)
            curr_file_EdList = genEditDistanceList(file_names[i])
            if len(q_file_EdList) <= len(curr_file_EdList):
                edTopDict[editdistance(q_file_EdList, curr_file_EdList)].append(
                    file_names[i])
            else:
                edTopDict[editdistance(curr_file_EdList, q_file_EdList)].append(
                    file_names[i])
    dists = []
    for k in edTopDict.keys():
        dists.append(k)
    dists.sort()
    topTenFilesList = []
    for i in range(len(dists)):
        topTenFilesList.extend(edTopDict[dists[i]])
    return edTopDict, topTenFilesList[:10]


def findDTWDistance(q_file):
    dtwTopDict = defaultdict(list)
    for i in range(len(file_names)):
        if q_file != file_names[i]:
            q_file_dtwList = genDtwDistanceList(q_file)
            curr_file_dtwList = genDtwDistanceList(file_names[i])
            if len(q_file_dtwList) <= len(curr_file_dtwList):
                dtwTopDict[dtw(np.array(q_file_dtwList), np.array(curr_file_dtwList), 'euclidean')].append(
                    file_names[i])
            else:
                dtwTopDict[dtw(np.array(curr_file_dtwList), np.array(q_file_dtwList), 'euclidean')].append(
                    file_names[i])
    dists = []
    for k in dtwTopDict.keys():
        dists.append(k)
    dists.sort()
    topTenFilesList = []
    for i in range(len(dists)):
        topTenFilesList.extend(dtwTopDict[dists[i]])
    return dtwTopDict, topTenFilesList[:10]


def genEditDistanceList(curr_file):
    edList = []

    tfw = open('output\\tf_txt\\tf_vectors_W_'+curr_file+'.wrd.txt', 'r')
    tfx = open('output\\tf_txt\\tf_vectors_X_'+curr_file+'.wrd.txt', 'r')
    tfy = open('output\\tf_txt\\tf_vectors_Y_'+curr_file+'.wrd.txt', 'r')
    tfz = open('output\\tf_txt\\tf_vectors_Z_'+curr_file+'.wrd.txt', 'r')

    tfw_dict = eval(tfw.read())
    str_list = []
    for k in tfw_dict.keys():
        temp = eval(k)[-1]
        temp_str = ""
        for i in range(len(temp)):
            temp_str += str(int(temp[i]))
        str_list.append(str(eval(k)[2]) + temp_str)
    edList.extend(str_list)

    tfx_dict = eval(tfx.read())
    str_list = []
    for k in tfx_dict.keys():
        temp = eval(k)[-1]
        temp_str = ""
        for i in range(len(temp)):
            temp_str += str(int(temp[i]))
        str_list.append(str(eval(k)[2]) + temp_str)
    edList.extend(str_list)

    tfy_dict = eval(tfy.read())
    str_list = []
    for k in tfy_dict.keys():
        temp = eval(k)[-1]
        temp_str = ""
        for i in range(len(temp)):
            temp_str += str(int(temp[i]))
        str_list.append(str(eval(k)[2]) + temp_str)
    edList.extend(str_list)

    tfz_dict = eval(tfz.read())
    str_list = []
    for k in tfz_dict.keys():
        temp = eval(k)[-1]
        temp_str = ""
        for i in range(len(temp)):
            temp_str += str(int(temp[i]))
        str_list.append(str(eval(k)[2]) + temp_str)
    edList.extend(str_list)
    return edList


def genDtwDistanceList(curr_file):
    dtwList = []

    winav_w = []
    winav_x = []
    winav_y = []
    winav_z = []

    dfw1 = pd.read_pickle("output\\W_" + curr_file + ".wrd")
    dfx1 = pd.read_pickle("output\\X_" + curr_file + ".wrd")
    dfy1 = pd.read_pickle("output\\Y_" + curr_file + ".wrd")
    dfz1 = pd.read_pickle("output\\Z_" + curr_file + ".wrd")

    for r in range(len(dfw1)):
        winav_w_row = 0
        winav_x_row = 0
        winav_y_row = 0
        winav_z_row = 0
        for c in range(len(dfw1.iloc[r])):
            winav_w_row += dfw1.iloc[r, c][-2]
            winav_x_row += dfw1.iloc[r, c][-2]
            winav_y_row += dfw1.iloc[r, c][-2]
            winav_z_row += dfw1.iloc[r, c][-2]
        winav_w.append(winav_w_row/len(dfw1.iloc[r]))
        winav_x.append(winav_x_row/len(dfw1.iloc[r]))
        winav_y.append(winav_y_row/len(dfw1.iloc[r]))
        winav_z.append(winav_z_row/len(dfw1.iloc[r]))

    dtwList = winav_w + winav_x + winav_y + winav_z

    return dtwList


def genDotProductList(curr_file, vector_model):
    dotProdList = []
    if vector_model == 1:
        tf_vectors_df = pd.read_csv("output\\all_tf_gestures.csv")
    else:
        tf_vectors_df = pd.read_csv("output\\all_tfidf_gestures.csv")
    file_list = list(tf_vectors_df.iloc[:, 0])
    file_list = [f.rstrip('.wrd') for f in file_list]
    file_dict = {x: index-1 for index, x in enumerate(file_list, start=1)}
    return tf_vectors_df.iloc[file_dict[curr_file], 1:]


def genEditDistanceFiles(file_names):
    distance_matrix = np.zeros((len(file_names), len(file_names)))
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i == j:
                distance_matrix[i][j] = 0
            elif distance_matrix[j][i] != 0:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                distance_matrix[i][j] = editdistance(genEditDistanceList(
                    file_names[i]), genEditDistanceList(file_names[j]))
    for i in range(len(distance_matrix)):
        print("\nWriting file "+str(i+1)+" of " + str(len(file_names))+"\n")
        edFileDict = defaultdict(list)
        for k, v in zip(distance_matrix[i], file_names):
            if v != file_names[i]:
                edFileDict[k].append(v)
        with open('output/task2/edit_dist_'+file_names[i]+'.txt', 'w') as f:
            print(edFileDict, file=f)


def genDTWDistanceFiles(file_names):
    distance_matrix = np.zeros((len(file_names), len(file_names)))
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i == j:
                distance_matrix[i][j] = 0
            elif distance_matrix[j][i] != 0:
                distance_matrix[i][j] = distance_matrix[j][i]
            else:
                distance_matrix[i][j] = dtw(np.array(genDtwDistanceList(
                    file_names[i])), np.array(genDtwDistanceList(file_names[j])), 'euclidean')
    for i in range(len(distance_matrix)):
        print("\nWriting file "+str(i+1)+" of " + str(len(file_names))+"\n")
        dtwFileDict = defaultdict(list)
        for k, v in zip(distance_matrix[i], file_names):
            if v != file_names[i]:
                dtwFileDict[k].append(v)
        with open('output/task2/dtw_dist_'+file_names[i]+'.txt', 'w') as f:
            print(dtwFileDict, file=f)


def editdistance(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n+1)
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 2
            current[j] = min(add, delete, change)

    return current[n]


def dtw(x, y, dist, window=1):
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, window + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    return D1[-1, -1]


if __name__ == "__main__":
    data_dir = input("Enter data directory name: ")
    mypath = data_dir + "\\W"
    Path("output/task2").mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlyfiles = [f for f in onlyfiles if f.endswith('.csv')]
    file_names = []
    for i in range(len(onlyfiles)):
        file_names.append(str(onlyfiles[i].replace(".csv", "")))

    print("The following files were found in the " + data_dir + " directory: \n")
    print(file_names)
    query_file = input("Enter query file name without extension: ")

    query_model = input("Enter the vector model 1 -> TF; 2-> TF-IDF: ")

    distance_metric = input(
        "Enter the distance metric that you want to use:\nOptions:\n1->Dot Product;\n2->PCA;\n3->SVD;\n4->NMF;\n5->LDA;\n6->Edit Distance;\n7->DTW")

    start = time.time()

    if distance_metric == "6":
        editTopDict, top10list = findEditDistance(query_file)
        print("\nThe Top 10 similar gestures to "+query_file +
              " based on  Edit Distance are as follows:\n")
        print(top10list)
        print("\nThe results in the form of <gesture,score> for "+query_file +
              " based on  Edit Distance are as follows:\n")
        ed_dList = list(editTopDict.keys())
        ed_dList.sort()
        for i in range(len(ed_dList)):
            if len(editTopDict[ed_dList[i]]) == 1:
                print("< "+str(editTopDict[ed_dList[i]][0]
                               )+","+str(ed_dList[i])+" >")
            else:
                for k in range(len(editTopDict[ed_dList[i]])):
                    print(
                        "< "+str(editTopDict[ed_dList[i]][k])+","+str(ed_dList[i])+" >")
        genEditDistanceFiles(file_names)
    elif distance_metric == "7":
        dtwqqDict, top10list_dtw = findDTWDistance(query_file)
        print("\nThe Top 10 similar gestures to "+query_file +
              " based on  DTW are as follows:\n")
        print(top10list_dtw)
        print("\nThe results in the form of <gesture,score> for "+query_file +
              " based on  DTW are as follows:\n")
        dtw_dList = list(dtwqqDict.keys())
        dtw_dList.sort()
        for i in range(len(dtw_dList)):
            if len(dtwqqDict[dtw_dList[i]]) == 1:
                print("< "+str(dtwqqDict[dtw_dList[i]][0]
                               )+","+str(dtw_dList[i])+" >")
            else:
                for k in range(len(dtwqqDict[dtw_dList[i]])):
                    print(
                        "< "+str(dtwqqDict[dtw_dList[i]][k])+","+str(dtw_dList[i])+" >")
        genDTWDistanceFiles(file_names)
    elif distance_metric == "1":
        if query_model == "1":
            dpqqDict, top10list_dp = findDotProductSimilarity(query_file, 1)
            print("\nThe Top 10 similar gestures to "+query_file +
                  " based on  Dot product are as follows:\n")
            print(top10list_dp)
            print("\nThe results in the form of <gesture,score> for "+query_file +
                  " based on  Dot product are as follows:\n")
            dp_dList = list(dpqqDict.keys())
            dp_dList.sort(reverse=True)
            for i in range(len(dp_dList)):
                if len(dpqqDict[dp_dList[i]]) == 1:
                    print("< "+str(dpqqDict[dp_dList[i]][0]
                                   )+","+str(dp_dList[i])+" >")
                else:
                    for k in range(len(dpqqDict[dp_dList[i]])):
                        print(
                            "< "+str(dpqqDict[dp_dList[i]][k])+","+str(dp_dList[i])+" >")
        elif query_model == "2":
            dpqqDict, top10list_dp = findDotProductSimilarity(query_file, 2)
            print("\nThe Top 10 similar gestures to "+query_file +
                  " based on  Dot product are as follows:\n")
            print(top10list_dp)
            print("\nThe results in the form of <gesture,score> for"+query_file +
                  " based on  Dot product are as follows:\n")
            dp_dList = list(dpqqDict.keys())
            dp_dList.sort(reverse=True)
            for i in range(len(dp_dList)):
                if len(dpqqDict[dp_dList[i]]) == 1:
                    print("< "+str(dpqqDict[dp_dList[i]][0]
                                   )+","+str(dp_dList[i])+" >")
                else:
                    for k in range(len(dpqqDict[dp_dList[i]])):
                        print(
                            "< "+str(dpqqDict[dp_dList[i]][k])+","+str(dp_dList[i])+" >")
        else:
            print("Enter a valid option")
    elif distance_metric == "2" or distance_metric == "3" or distance_metric == "4" or distance_metric == "5":
        useDimReductionAlgs(query_file, query_model, distance_metric)
    else:
        print("Enter a valid option")

    end = time.time()
    print("\nRun time: "+str(end-start)+" secs")
