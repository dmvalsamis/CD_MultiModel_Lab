#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 12:02:38 2023

@author: aleoikon
"""
import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import glob
import os
from changedetection.utils.visualize import create_rgb

def read_rasters(path):
    im_names = glob.glob(os.path.join(path, '*B04.tif'))
    r = io.imread(im_names[0])
    im_names = glob.glob(os.path.join(path, '*B03.tif'))
    g = io.imread(im_names[0])
    im_names = glob.glob(os.path.join(path, '*B02.tif'))
    b = io.imread(im_names[0])
    I = np.stack((r, g, b), axis=2).astype('float')
    return I

def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f_m = f1_score(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    
    return cm, acc, spec, rec, prec, f_m

def change_detection(dataset_name, input_rasters, pred_path):

    # Load images
    imgs = []
    for timeframe in input_rasters:
        imgs.append(read_rasters(timeframe))
    img1_or = imgs[0]
    img2_or = imgs[1]
    
    # Load predictions
    pred_file = os.path.join(pred_path, f'{dataset_name}_conv_cm.npy')
    y_pred_conv = np.load(pred_file)
    
    # Load ground truth
    cm_path = f'/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}/cm/cm.tif'
    y_true = io.imread(cm_path)
    
    # Compute metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_conv.flatten()
    
    cm, acc, spec, rec, prec, f_m = get_metrics(y_true_flat, y_pred_flat)
    
    # Print metrics
    print(f"Dataset: {dataset_name}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc}")
    print(f"Specificity: {spec}")
    print(f"Recall: {rec}")
    print(f"Precision: {prec}")
    print(f"F-measure: {f_m}")
    
    # fig, ax = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
    # font = 20
    # ax[0].imshow(create_rgb(img1_or))
    # ax[0].set_title('t1', fontsize=font)
    # ax[0].axis('off')
    # ax[1].imshow(create_rgb(img2_or))
    # ax[1].set_title('t2', fontsize=font)
    # ax[1].axis('off')
    # ax[2].imshow(y_pred_conv, cmap='gray')
    # ax[2].set_title('Conv Prediction', fontsize=font)
    # ax[2].axis('off')
    # plt.show()

# Read the dataset names from the all.txt file
dataset_names_file = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/all.txt'

with open(dataset_names_file, 'r') as file:
    dataset_names_line = file.readline().strip()  # Read the line containing all dataset names
    dataset_names = dataset_names_line.split(',')  # Split the line into individual dataset names

# Iterate over dataset names
for dataset_name in dataset_names:
    try:
        input_rasters = [
            f'/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}/img1_cropped/',
            f'/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}/img2_cropped/'
        ]
        pred_path = f'/data/valsamis_data/data/CBMI/CBMI_0.3/Predictions/Depth_2/Trainable_ASPP/Conv/{dataset_name}'
        change_detection(dataset_name, input_rasters, pred_path)
    except Exception as e:
        print(f"Error occurred in dataset: {dataset_name}")
        print(f"Error message: {str(e)}")

print("Done")
