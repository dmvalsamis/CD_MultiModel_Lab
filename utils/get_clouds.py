#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:39:01 2022

@author: aleoikon
"""

import pandas as pd
import numpy as np

#s2mtcp_target = '/home/aleoikon/Documents/data/ssl/s2mtcp/patches/task11238/'

s2mtcp_target = '/home/aleoikon/Documents/data/ssl/s2mtcp/patches_colorshifted/task1/'

df = pd.read_csv(s2mtcp_target+'dataset.csv', dtype=str)

count_left = 0
count_right = 0
index_left =[]
index_right=[]
clouded_index =[]
for index in range(len(df)):
    img1 = np.load(s2mtcp_target+df['pair1'][index])
    img2 = np.load(s2mtcp_target+df['pair2'][index])
    if np.max(img1[:,:,13]) != 0:
        index_left.append(index)
        count_left+=1
        
    if np.max(img2[:,:,13]) != 0:
        index_right.append(index)
        count_right+=1
    
    if np.max(img1[:,:,13]) != 0 or np.max(img2[:,:,13]) != 0:
        clouded_index.append(index)


to_Drop = np.array(clouded_index)

uncloud_df = df.drop(to_Drop)
uncloud_df.to_csv(s2mtcp_target+'dataset_unclouded.csv')


img = np.load(s2mtcp_target+df['pair2'][14004])
import matplotlib.pyplot as plt
plt.imshow(img[:,:,13])
np.max(img[:,:,13])!=0