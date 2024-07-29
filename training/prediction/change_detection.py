#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 12:02:38 2023

@author: aleoikon
"""
import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import numpy as np
from architectures.conv_classifier import conv_classifier_two
from architectures.branch import branch_cva
from utils.layer_select import feature_selector_cva
from architectures.fusion_maria import fusion
import matplotlib.pyplot as plt
from utils.visualize import create_rgb
from numpy import expand_dims
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import remove_small_objects
import glob
import os
from skimage import io

#hide the gpus for timing the application
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def calculate_distancemap(f1, f2):
    """
    calcualtes pixelwise euclidean distance between images with multiple imput channels

    Parameters
    ----------
    f1 : np.ndarray of shape (N,M,D)
        image 1 with the channels in the third dimension 
    f2 : np.ndarray of shape (N,M,D)
        image 2 with the channels in the third dimension 
        
    Returns
    -------
    np.ndarray of shape(N,M)
        pixelwise euclidean distance between image 1 and image 2

    """
    dist_per_fmap= [(f2[i,:,:]-f1[i,:,:])**2 for i in range(f1.shape[0])]
    
    return np.sqrt(sum(dist_per_fmap))

def read_rasters(path): # Diavazi tis 3s bandes kai ftiaxni mia nea rbg eikona
    path = '/data/aleoikon_data/change_detection/ssl/CBMI_Dataset/Massarosa/img2_cropped'
    im_names = glob.glob(os.path.join(path, '*B04.tif')) # search for files with names containing 'B04.tif'
    r = io.imread(im_names[0])
    print(im_names)
    im_names = glob.glob(os.path.join(path, '*B03.tif'))  # search for files with names containing 'B03.tif'
    g = io.imread(im_names[0])
    print(im_names)
    im_names = glob.glob(os.path.join(path, '*B02.tif'))  # search for files with names containing 'B02.tif'
    b = io.imread(im_names[0])
    print(im_names)
    I = np.stack((r,g,b),axis=2).astype('float')
    return I


def change_detection(dataset_name, input_rasters):
    # Load images
    imgs = []
    for timeframe in input_rasters:
        imgs.append(read_rasters(timeframe))
    img1_or = imgs[0]
    img2_or = imgs[1]

    path = f'/home/dvalsamis/Documents/data/outputs/cd/{dataset_name}'
    # Load change detection model
    saved_model = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_Onera_5bfd.h5'
    
    shape = img1_or.shape
    print("The shape is : ", shape)
    depth = 2
    dropout = 0.1
    decay = 0.0001
    ImageSize_X = shape[0]
    ImageSize_Y = shape[1]
    n_ch = shape[2]
    
    cd_model = conv_classifier_two(depth, dropout, decay, ImageSize_X, ImageSize_Y, n_ch)
    cd_model.load_weights(saved_model)
    
    #predictions
    img1 = expand_dims(img1_or, axis=0)
    img1 = (img1 - img1.mean()) / img1.std()
    img2 = expand_dims(img2_or, axis=0)
    img2 = (img2 - img2.mean()) / img2.std()
    
    #conv predictions
    prediction = cd_model.predict([img1,img2])
    y_preds = np.argmax(prediction[0], axis=2)
    y_pred_conv = y_preds
    fig, ax = plt.subplots(1, 3, figsize=(20,10), constrained_layout=True)
    font=20
    #create subplots
    ax[0].imshow(create_rgb(img1_or))
    ax[0].set_title('t1', fontsize=font)
    ax[0].axis('off')
    ax[1].imshow(create_rgb(img2_or))
    ax[1].set_title('t2', fontsize=font)
    ax[1].axis('off')
    ax[2].imshow(y_pred_conv,cmap='gray')
    ax[2].set_title('Conv Prediction', fontsize=font)
    ax[2].axis('off')
    #Otsu & triangle
    branch_model = branch_cva(dropout, decay, depth, ImageSize_X,ImageSize_Y,n_ch)
    branch_model = feature_selector_cva(depth, cd_model, branch_model) 
    
    feature_maps_left = branch_model.predict(img1[:,:,:,0:3])
    feature_maps_right = branch_model.predict(img2[:,:,:,0:3])
    
    left = np.ndarray(shape=(32,ImageSize_X,ImageSize_Y))
    right = np.ndarray(shape=(32,ImageSize_X,ImageSize_Y))
    for i in range(left.shape[0]):
        left[i] = feature_maps_left[0,:,:,i]
        right[i] = feature_maps_right[0,:,:,i]
            
    distmap = calculate_distancemap(left, right)
    
    binary_otsu = distmap > threshold_otsu(distmap)
    binary_otsu = remove_small_objects(binary_otsu,min_size=55)
    
    binary_triangle = distmap > threshold_triangle(distmap)
    binary_triangle = remove_small_objects(binary_triangle,min_size=55)
    
    y_pred_otsu=binary_otsu
    y_pred_triangle=binary_triangle
    
    param = 3.5
    fused_mask = fusion(y_pred_triangle,y_pred_otsu,y_pred_conv, param)
    y_pred_fusion = fused_mask
    
    fig, ax = plt.subplots(1, 3, figsize=(20,10), constrained_layout=True)
    font=20
    #create subplots
    ax[0].imshow(create_rgb(img1_or))
    ax[0].set_title('t1', fontsize=font)
    ax[0].axis('off')
    ax[1].imshow(create_rgb(img2_or))
    ax[1].set_title('t2', fontsize=font)
    ax[1].axis('off')
    ax[2].imshow(y_pred_fusion,cmap='gray')
    ax[2].set_title('Fusion Prediction', fontsize=font)
    ax[2].axis('off')
    
    
    # Check if the folder exists
    if not os.path.exists(path):
        # Create the folder
        os.makedirs(path)
    plt.savefig(os.path.join(path, f'{dataset_name}_fusion_cm.png'))
    np.save(os.path.join(path, f'{dataset_name}_fusion_cm.npy'), y_pred_fusion)
    return y_pred_fusion

#npy_path = '/home/aleoikon/Documents/data/puc4_demo/npys/'
#input_rasters = ['/home/aleoikon/Documents/projects/KR06 integration/inputs/input_rasters/factory/20230528T100559/', '/home/aleoikon/Documents/projects/KR06 integration/inputs/input_rasters/factory/20230103T101411/']
dataset_name = 'aguasclaras'
input_rasters = [
    '/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images/{dataset_name}/imgs_1',
    '/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images/{dataset_name}/imgs_2'
]
change_mask = change_detection(dataset_name, input_rasters)

print("Done")