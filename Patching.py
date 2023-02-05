#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:57:14 2023

@author: Silvana
"""

import numpy as np
import pandas as pd

# function to identify the categories
def unique(list):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            
    return(unique_list)

# function to divide the spectrum into 16X16 overlapping patches
def patching(spec):

    Nx = spec.shape[0]
    Ny = spec.shape[1]
    # =================== COMPUTING PATCHING Y-DIMENSION ===================
    # compute overlapping intervals for patching
    step = 16
    uplim = step
    center_y = step/2
    while uplim < Ny:
        olap = uplim - 6 #computing the overlap
        uplim = olap + step
        cent = uplim - step/2
        center_y = np.vstack((center_y,cent))
    # removing last element, values outside range
    center_y = center_y[:-1]
  
    # =================== COMPUTING PATCHING X-DIMENSION ===================
    # compute overlapping intervals for patching
    Nx = 498
    step = 16
    uplim = step
    center_x = step/2
    while uplim < Nx:
        olap = uplim - 6 #computing the overlap
        uplim = olap + step
        cent = uplim - step/2
        center_x = np.vstack((center_x,cent))
    # removing last element, values outside range
    center_x = center_x[:-1]
    
    # =================== PATCHING AND FLATTENING ===================
    patches = []
    for ptx in range(center_x.size):
        ax = (center_x[ptx][0] - step/2).astype(int) 
        bx = (center_x[ptx][0] + step/2).astype(int)  
        for pty in range(center_y.size): 
            ay = (center_y[pty][0] - step/2).astype(int)  
            by = (center_y[pty][0] + step/2).astype(int) 
            patch = np.matrix.flatten(spec[ax:bx,:][:,ay:by])
            # patches flattened to a 1D array
            # the stacking is performed bottom-up
            if pty == 0 and ptx == 0:
                patches = patch
            else:
                patches = np.column_stack((patches,patch))
    return(patches)

# load dataset names and categories
file_csv = pd.read_csv('esc50-dataset/esc50.csv')
names = file_csv['filename']
categories = file_csv['category']

labels = unique(categories)
# variable to store the labels
label = -1
for ind in range(2000):
  word = categories[ind]
  label = np.vstack((label,labels.index(word))).astype(int)

label = np.delete(label,0) 

for ind in range(2000):
  data = np.load('Preprocessed/'+names[ind]+'.npz')
  spec=data['fbanks']
  patches = patching(spec)
  # save data individually in folder Patches
  np.savez('Patches/' +names[ind]+ '.npz',patches=patches)