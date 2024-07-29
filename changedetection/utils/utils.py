#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:50:06 2022

@author: aleoikon
"""

import numpy as np
from scipy.ndimage import zoom

def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    # crop if necesary
    I = I[:s[0],:s[1]]
    si = I.shape

    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])

    return np.pad(I,((0,p0),(0,p1)),'edge')

def feature_scaling(img, method):
    I = img
    if method == "STAND":
        I = (I - I.mean()) / I.std()
        return I
    if method == "MINMAX":
        I = ((I - np.nanmin(I))/(np.nanmax(I) - np.nanmin(I)))
        return I
    else:
        return I

def create_rgb(x,channel):
    if channel == 'red':
        r = x[:,:,1]
        r = np.expand_dims(r, axis=2)
        return r
    if channel == 'green':
        g = x[:,:,2]
        g = np.expand_dims(g, axis=2)
        return g
    if channel == 'blue':
        b  = x[:,:,3]
        b = np.expand_dims(b, axis=2)
        return b
    if channel == 'rgb':
        r = x[:,:,1]
        g = x[:,:,2]
        b  = x[:,:,3]
        rgb = np.dstack((r,g,b))
        return(rgb)
    if channel == 'rgbvnir':
        r = x[:,:,1]
        g = x[:,:,2]
        b  = x[:,:,3]
        vnir = x[:,:,8]
        rgbvnir = np.stack((r,g,b,vnir),axis=2).astype('float')
        #rgb = np.dstack((r,g,b))
        return(rgbvnir)
    if channel == 'eq20':
        r = x[:,:,1]
        s = r.shape
        ir1 = adjust_shape(zoom(x[:,:,4],2),s)
        ir2 = adjust_shape(zoom(x[:,:,5],2),s)
        ir3 = adjust_shape(zoom(x[:,:,6],2),s)
        nir2 = adjust_shape(zoom(x[:,:,8],2),s)
        swir2 = adjust_shape(zoom(x[:,:,11],2),s)
        swir3 = adjust_shape(zoom(x[:,:,12],2),s)
        x = np.stack((ir1,ir2,ir3,nir2,swir2,swir3),axis=2).astype('float') 
        return x
    else:
        return x
        print("NOT CORRECT CHANNELS")