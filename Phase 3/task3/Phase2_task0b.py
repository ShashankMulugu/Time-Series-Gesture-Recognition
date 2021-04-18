#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 18:58:41 2020

@author: mageshsridhar
"""

import pandas as pd
import os
import math
import pickle
import re
import copy
import time


all_words = {}
all_words_with_filenames = {}
filenames = []


def vocabulary(N, file_dir):
    for filename in sorted(os.listdir(file_dir)):
        if filename.endswith(".wrd"):
            N += 1
            file = open(file_dir+'/'+filename, 'rb')
            temp_df = pickle.load(file)
            file.close()
            filenames.append(filename)
            for i in range(len(temp_df)):
                for j in range(len(temp_df.iloc[0])):
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(1)
                    all_words[str(temp_df.iloc[i, j])] = 0
    return N


def tf_calculation(file_dir):
    tf_all_words_all_files = pd.DataFrame()
    all_df = pd.DataFrame()
    for filename in sorted(os.listdir(file_dir)):
        if filename.endswith(".wrd"):
            file = open(file_dir+'/'+filename, 'rb')
            # dump information to that file
            temp_df = pickle.load(file)
            # close the file
            file.close()
            for i in range(len(temp_df)):
                for j in range(len(temp_df.iloc[0])):
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
            all_df = pd.concat([all_df, temp_df], axis=1)
            words_count_in_rows = {}
            for i in range(len(temp_df)):
                for j in range(len(temp_df.iloc[0])):
                    if str(temp_df.iloc[i, j]) in words_count_in_rows:
                        words_count_in_rows[str(temp_df.iloc[i, j])] += 1
                    else:
                        words_count_in_rows[str(temp_df.iloc[i, j])] = 1
            tf_in_file = {}
            for i in words_count_in_rows.keys():
                tf_in_file[i] = words_count_in_rows[i]/len(temp_df.iloc[0])
            tf_file = []
            for i in range(len(temp_df)):
                temp_tf_row = []
                for j in range(len(temp_df.iloc[0])):
                    temp_tf_row.append(tf_in_file[str(temp_df.iloc[i, j])])
                tf_file.append(temp_tf_row)
            tf_file = pd.DataFrame(tf_file)
            tf_file.to_csv(file_dir+'/tf/'+filename.replace('.wrd',
                                                            '_tf')+".csv", header="False", index="False", sep=",")
            tf_all_words_per_file = copy.deepcopy(all_words)
            for i in tf_in_file.keys():
                tf_all_words_per_file[re.sub(
                    r"\,\s\'.{1,7}\.\w{3}\'", "", i)] = tf_in_file[i]
            f = open(file_dir+"/tf_txt/"+"tf_vectors_"+filename+".txt", "a")
            f.write(str(tf_in_file))
            tf_all_words_all_files = pd.concat(
                [tf_all_words_all_files, pd.DataFrame(tf_all_words_per_file, index=[0])], axis=0)
    tf_all_words_all_files.to_csv(
        file_dir+"/all_tf.csv", header="True", index="True", sep=",")
    return all_df


def tfidf_calculation(all_df, file_dir, N):
    n = N
    tfidf_all_words_all_files = pd.DataFrame()
    words_in_files = {}
    filenames = []
    for i in range(len(all_df)):
        for j in range(len(all_df.iloc[0])):
            word = []
            word.append(eval(all_df.iloc[i, j])[0])
            word.append(eval(all_df.iloc[i, j])[2])
            word.append(eval(all_df.iloc[i, j])[3])
            word = str(word)
            if word in words_in_files:
                words_in_files[word].append(eval(all_df.iloc[i, j])[1])
            else:
                words_in_files[word] = [eval(all_df.iloc[i, j])[1]]
    for i in words_in_files.keys():
        words_in_files[i] = len(set(words_in_files[i]))
    for filename in sorted(os.listdir(file_dir)):
        if filename.endswith(".wrd"):
            filenames.append(filename)
            file = open(file_dir+'/'+filename, 'rb')
            temp_df = pickle.load(file)
            temp_tf_df = pd.read_csv(
                file_dir+'/tf/'+filename.replace('.wrd', '_tf')+".csv", header=0, index_col=0, sep=",")
            file.close()
            tfidf_in_file = {}
            tfidf_all_words_per_file = copy.deepcopy(all_words)
            temp_df2 = copy.deepcopy(temp_df)
            for i in range(len(temp_df)):
                for j in range(len(temp_df.iloc[0])):
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(3)
                    temp_df.iloc[i, j].pop(1)
                    word = temp_df.iloc[i, j]
                    temp_df2.iloc[i, j] = temp_tf_df.iloc[i, j] * \
                        math.log(n/words_in_files[str(word)])
                    tfidf_in_file[str(word)] = temp_df2.iloc[i, j]
                    tfidf_all_words_per_file[str(word)] = temp_df2.iloc[i, j]
            temp_df2.to_csv(file_dir+'/tfidf/'+filename.replace(".wrd",
                                                                '_tfidf')+".csv", header="False", index="False", sep=",")
            f = open(file_dir+"/tfidf_txt/" +
                     "tfidf_vectors_"+filename+".txt", "a")
            f.write(str(tfidf_in_file))
            tfidf_all_words_all_files = pd.concat([tfidf_all_words_all_files, pd.DataFrame(
                tfidf_all_words_per_file, index=[0])], axis=0)
    tfidf_all_words_all_files.to_csv(
        file_dir+"/all_tfidf.csv", header="True", index="True", sep=",")


def combine_tf_gestures(file_dir):
    filenames = []
    all_tf = pd.read_csv(file_dir+"/all_tf.csv",
                         header=0, index_col=0, sep=',')
    for filename in sorted(os.listdir(file_dir)):
        if filename.startswith("W"):
            filenames.append(filename.replace('W_', ''))
    n = len(filenames)
    for i in range(len(filenames)):
        all_tf.iloc[i] += all_tf.iloc[i+n]
        all_tf.iloc[i] += all_tf.iloc[i+n*2]
        all_tf.iloc[i] += all_tf.iloc[i+n*3]
    all_tf = all_tf.take(list(range(n)))
    all_tf.index = filenames
    all_tf.to_csv(file_dir+"/all_tf_gestures.csv",
                  header="True", index="True", sep=",")
    file_gesture = open(file_dir+"/vectors/tf_vectors_" +
                        filename.replace("tf_vectors_W_", '')+".txt", "a+")
    for filename in sorted(os.listdir(file_dir+"/tf_txt")):
        if filename.startswith("tf_vectors_W"):
            fW = open(file_dir+"/tf_txt/"+filename, "r")
            file_gesture.write(fW.read())
            fX = open(file_dir+"/tf_txt/"+filename.replace('W', 'X'), "r")
            file_gesture.write(fX.read())
            fY = open(file_dir+"/tf_txt/"+filename.replace('W', 'Y'), "r")
            file_gesture.write(fY.read())
            fZ = open(file_dir+"/tf_txt/"+filename.replace('W', 'Z'), "r")
            file_gesture.write(fZ.read())


def combine_tfidf_gestures(file_dir):
    filenames = []
    all_tfidf = pd.read_csv(file_dir+"/all_tfidf.csv",
                            header=0, index_col=0, sep=',')
    for filename in sorted(os.listdir(file_dir)):
        if filename.startswith("W"):
            filenames.append(filename.replace('W_', ''))
    n = len(filenames)
    for i in range(len(filenames)):
        all_tfidf.iloc[i] += all_tfidf.iloc[i+n]
        all_tfidf.iloc[i] += all_tfidf.iloc[i+n*2]
        all_tfidf.iloc[i] += all_tfidf.iloc[i+n*3]
    all_tfidf = all_tfidf.take(list(range(n)))
    all_tfidf.index = filenames
    all_tfidf.to_csv(file_dir+"/all_tfidf_gestures.csv",
                     header="True", index="True", sep=",")
    file_gesture = open(file_dir+"/vectors/tfidf_vectors_" +
                        filename.replace("tfidf_vectors_W_", '')+".txt", "a+")
    for filename in sorted(os.listdir(file_dir+"/tfidf_txt")):
        if filename.startswith("tfidf_vectors_W"):
            fW = open(file_dir+"/tfidf_txt/"+filename, "r")
            file_gesture.write(fW.read())
            fX = open(file_dir+"/tfidf_txt/"+filename.replace('W', 'X'), "r")
            file_gesture.write(fX.read())
            fY = open(file_dir+"/tfidf_txt/"+filename.replace('W', 'Y'), "r")
            file_gesture.write(fY.read())
            fZ = open(file_dir+"/tfidf_txt/"+filename.replace('W', 'Z'), "r")
            file_gesture.write(fZ.read())


if __name__ == "__main__":
    N = 0
    start = time.time()
    file_dir = input("Enter the file directory: ")
    os.mkdir(file_dir+"/vectors")
    # creating tf folder to store CSVs with TF values
    os.mkdir(file_dir+"/tf/")
    os.mkdir(file_dir+"/tf_txt/")
    # creating tfidf folder to store CSVs with TFIDF values
    os.mkdir(file_dir+"/tfidf/")
    # creating idf folder to store CSVs with IDF values
    os.mkdir(file_dir+"/tfidf_txt/")
    N = vocabulary(N, file_dir)
    all_df = tf_calculation(file_dir)
    all_df.columns = range(all_df.shape[1])
    all_df.to_csv(file_dir+"/concatenated.csv",
                  header="False", index="False", sep=",")
    all_df = pd.read_csv(file_dir+"/concatenated.csv",
                         header=0, index_col=0, sep=",")
    tfidf_calculation(all_df, file_dir, N)
    combine_tf_gestures(file_dir)
    combine_tfidf_gestures(file_dir)
    end = time.time()
    print("Run time:"+str(end-start))
