#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:27:02 2020

@author: mageshsridhar
"""

import pandas as pd


df = pd.read_csv("output/SVDScores.csv",header=0,index_col=0,sep=',')
maxi = list(df.max())
filenames = df.columns.values.tolist()
partitions= {}
for i in range(len(df)):
    partitions[i] = []
for i in range(len(maxi)):
    partitions[list(df[filenames[i]]).index(maxi[i])].append(filenames[i])
for i in range(len(partitions)):
    print("Partition "+str(i+1)+":"+str(partitions[i]))

    