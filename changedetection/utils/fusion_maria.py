#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:27:17 2022

@author: Maria Pegia
"""

import numpy as np



def fusion(triangle_cm, otsu_cm, conv_cm, param):
    triangle_cm = triangle_cm.astype(int)
    otsu_cm = otsu_cm.astype(int)
    conv_cm = conv_cm.astype(int)
    
    cm = np.zeros(conv_cm.shape)

    sim = np.bitwise_and(triangle_cm, otsu_cm)  # Keep similarities

    rows = sim.shape[0]
    for i in range(rows):
        x = conv_cm[i, :]
        y = sim[i, :]
        dist = np.linalg.norm(x - y)

        cm[i, :] = x if dist > param else y

    return cm