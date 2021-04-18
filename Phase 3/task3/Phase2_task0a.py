# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:20:27 2020

@author: Chandu
"""

import scipy
import numpy as np
import pandas as pd
from scipy import stats
from scipy import integrate
from scipy.spatial import distance
import glob
import os
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sb
import heapq
import fnmatch
from functools import reduce

#data = pd.read_csv("C:/Users/Chandu/Documents/Academics/Sem_3/MWDB/Project_1/Phase1_sample_data/Z/1.csv", header=None)

path = input("Enter the input files directory path : ")
print("Your directory is:" + path)
w = input("Enter Window length : ")
print("Your window lenght is :" + w)
s = input("Enter shift length : ")
print("Your shift length is :" + s)
r = input("Enter resolution : ")
print("Your resolutions is :" + r)

w = int(w)
s = int(s)
r = int(r)


def getMean(data):
    return data.mean(axis=1)


def getStandardDeviation(data):
    return data.std(axis=1)

# print(std)

# **************** Normalize the data between -1 to 1 **************


def normalizeData(data):
    data1 = data.copy()
    for index, row in data1.iterrows():
        mini = min(row)
        maxi = max(row)
        col = 0
        if mini == maxi:
            for val in row:
                data1.iloc[index, col] = 0
                col += 1
        else:
            for val in row:
                val = 2*(val - mini)/(maxi - mini) - 1
                data1.iloc[index, col] = val
                col += 1
    return data1

#normalized_data = normalizeData(data)

# ***************** Create bands from 1 to 2r ***********************


def createbands(r):
    mu = 0
    sigma = 0.25
    def gaussian(x): return np.exp(-((x-mu)**2)/(2*sigma*sigma))
    deno = integrate.quad(gaussian, -1, 1)
    k = deno[0]-deno[1]
    previous_range = -1
    bands = []

    for i in range(1, r+1):
        if i == r:
            bands.append(0)
            continue

        b = (i-r)/r
        a = (i-r-1)/r
        integral = integrate.quad(gaussian, a, b)
        current_range = 2*(integral[0] - integral[1])/k
        previous_range += current_range
        bands.append(previous_range)

    for i in range(len(bands)-2, -1, -1):
        bands.append(abs(bands[i]))
    bands.append(1)

    return bands


# print(bands)

# ************* Quantize the normalized data from 1 to 2r ******************
def quantizeData(data, r):
    bands = createbands(r)
    data1 = data.copy()
    for index, row in data1.iterrows():
        col = 0
        for val in row:
            for i in range(0, 2*r):
                if val <= bands[i]:
                    data1.iloc[index, col] = int(i+1)
                    break
            col += 1
    return data1
#quantized_data = quantizeData(normalized_data, r)


# ************ Create window data ****************************************

def createWindows(quantized_data, nomralized_data, window_length, shift_length, filename, mean, standardDeviation, component_name):
    window_data = pd.DataFrame()
    w = window_length
    s = shift_length
    data = quantized_data.copy()
    for index, row in data.iterrows():

        t = 0
        current_row = []
        while t+w < len(row):
            l2 = []
            l2.append(component_name)
            l2.append(filename)
            l2.append(index+1)
            l2.append(t)
            l2.append(mean[index])
            l2.append(standardDeviation[index])
            new_list = []
            normalized_windows = []
            for i in range(t, t+w):
                new_list.append(data.iloc[index, i])
                normalized_windows.append(normalized_data.iloc[index, i])

            win_avg_amplitude = reduce(
                lambda a, b: a + b, normalized_windows) / len(normalized_windows)

            l2.append(win_avg_amplitude)
            l2.append(new_list)
            current_row.append(l2)
            t = t+s
        temp_df = pd.DataFrame([current_row])
        window_data = window_data.append(temp_df)
    return window_data

#filename = '1.csv'
#window_data = createWindows(quantized_data, normalized_data, w, s, filename, m, std)
# print(window_data.iloc[0,1])
#np.savetxt("output/1.wrd", window_data.values, fmt='%s', delimiter=",")


# **************Run task0a - creating .wrd files for all components *****
#path = 'C:/Users/Chandu/Documents/Academics/Sem_3/MWDB/Project_2/Phase2_sample_data/data'
for root, dirnames, filenames in os.walk(path):
    print(dirnames)
    print(root)
    x = root.replace('\\', '/')
    component_name = root.split('\\')[-1].split('/')[-1]
    count = 0
    for filename in fnmatch.filter(filenames, '*.csv'):
        filepath = x + '/' + filename
        data = pd.read_csv(filepath, header=None)
        filename = (filename.split('/')[-1]).split('\\')[-1]
        output_file = filename.split('.')[0]
        output_file += '.wrd'
        output_file = component_name + '_' + output_file
        output_file_path = "./output/" + output_file

        # *************** Run all funtions in order ******
        m = getMean(data)
        std = getStandardDeviation(data)
        normalized_data = normalizeData(data)
        quantized_data = quantizeData(normalized_data, r)
        window_data = createWindows(
            quantized_data, normalized_data, w, s, filename, m, std, component_name)
        # *******************

        window_data.to_pickle(output_file_path)
        count += 1
        print("files complete: " + component_name + '_'+filename)

# C:\Users\Chandu\Documents\Academics\Sem_3\MWDB\Project_2\Phase2_sample_data\data
