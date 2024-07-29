#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 12:02:38 2023

@author: aleoikon
"""
import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import numpy as np
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp,conv_classifier_two_with_aspp
from architectures.branch import branch_cva, branch_cva_with_nspp, two_branch_cva_with_aspp
from changedetection.utils.layer_select import feature_selector_cva, feature_selector_cva_with_nspp, two_feature_selector_cva_aspp
from changedetection.utils.fusion_maria import fusion
import matplotlib.pyplot as plt
from changedetection.utils.visualize import create_rgb
from numpy import expand_dims
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.morphology import remove_small_objects
import glob
import os
from skimage import io

#hide the gpus for timing the application
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def calculate_distancemap(f1, f2):

    dist_per_fmap= [(f2[i,:,:]-f1[i,:,:])**2 for i in range(f1.shape[0])]
    
    return np.sqrt(sum(dist_per_fmap))

def read_rasters(path):
    im_names = glob.glob(os.path.join(path,'*B04.tif'))  # search for files with names containing 'B04.tif'
    print(im_names)
    r = io.imread(im_names[0])
    print(im_names)
    im_names = glob.glob(os.path.join(path,'*B03.tif'))  # search for files with names containing 'B03.tif'
    g = io.imread(im_names[0])
    print(im_names)
    im_names = glob.glob(os.path.join(path,'*B02.tif'))  # search for files with names containing 'B02.tif'
    b = io.imread(im_names[0])
    print(im_names)
    I = np.stack((r,g,b),axis=2).astype('float')
    return I


def change_detection(dataset_name, input_rasters):


    # Load images
    imgs = []
    for timeframe in input_rasters:
        print(timeframe)
        imgs.append(read_rasters(timeframe))
    img1_or = imgs[0]
    img2_or = imgs[1]



    # Load change detection model
    path = f'/home/dvalsamis/Documents/data/Onera/Predictions/Depth_2/Trainable_Onera/{dataset_name}'
    # Load change detection model
    saved_model = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_Onera_0a8d.h5'         
    

    # Check if the folder exists
    if not os.path.exists(path):
        # Create the folder
        os.makedirs(path)

    shape = img1_or.shape
    depth = 2
    dropout = 0.1
    decay = 0.0001
    ImageSize_X = shape[0]
    ImageSize_Y = shape[1]
    n_ch = shape[2]
    
    cd_model = conv_classifier_two_with_aspp(depth, dropout, decay, ImageSize_X, ImageSize_Y, n_ch)
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
    
    plt.savefig(os.path.join(path, f'{dataset_name}_conv_cm.png'))
    np.save(os.path.join(path, f'{dataset_name}_conv_cm.npy'), y_pred_conv)
    
    #Otsu & triangle
    branch_model = two_branch_cva_with_aspp(dropout, decay, depth, ImageSize_X,ImageSize_Y,n_ch)
    branch_model = two_feature_selector_cva_aspp(depth, cd_model, branch_model) 
    
    # Get the output from the model, which is (1, 96, 96, 32)
    combined_features = branch_model.predict([img1, img2])

    # Calculate the mean across the channels to get a (96, 96) map
    mean_feature_map = np.mean(combined_features, axis=-1)[0]
    
    # Otsu's thresholding
    binary_otsu = mean_feature_map > threshold_otsu(mean_feature_map)
    binary_otsu = remove_small_objects(binary_otsu, min_size=100)

    # Triangle thresholding
    binary_triangle = mean_feature_map > threshold_triangle(mean_feature_map)
    binary_triangle = remove_small_objects(binary_triangle, min_size=100)

    y_pred_otsu = binary_otsu
    y_pred_triangle = binary_triangle

    #Otsu outputs------------------------------------------------------------------------------------------
    
    fig, ax = plt.subplots(1, 3, figsize=(20,10), constrained_layout=True)
    font=20
    #create subplots
    ax[0].imshow(create_rgb(img1_or))
    ax[0].set_title('t1', fontsize=font)
    ax[0].axis('off')
    ax[1].imshow(create_rgb(img2_or))
    ax[1].set_title('t2', fontsize=font)
    ax[1].axis('off')
    ax[2].imshow(y_pred_otsu,cmap='gray')
    ax[2].set_title('Otsu Prediction', fontsize=font)
    ax[2].axis('off')
    
    plt.savefig(os.path.join(path, f'{dataset_name}_otsu_cm.png'))
    np.save(os.path.join(path, f'{dataset_name}_otsu_cm.npy'), y_pred_otsu)

    #Triangle Outputs------------------------------------------------------------------------------------------
    
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
    ax[2].set_title('Triangle Prediction', fontsize=font)
    ax[2].axis('off')
    
    plt.savefig(os.path.join(path, f'{dataset_name}_triangle_cm.png'))
    np.save(os.path.join(path, f'{dataset_name}_triangle_cm.npy'), y_pred_triangle)


    #Conv Outputs------------------------------------------------------------------------------------------

    param = 3.25
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
 
    plt.savefig(os.path.join(path, f'{dataset_name}_fusion_cm.png'))
    np.save(os.path.join(path, f'{dataset_name}_fusion_cm.npy'), y_pred_fusion)
    return y_pred_fusion




# Read the dataset names from the all.txt file

dataset_names_file = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI 0.3_initial/all.txt'

dataset_names_file =  '/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images/all.txt'


with open(dataset_names_file, 'r') as file:
    dataset_names_line = file.readline().strip()  # Read the line containing all dataset names
    dataset_names = dataset_names_line.split(',')  # Split the line into individual dataset names

# Iterate over dataset names
for dataset_name in dataset_names:
    try:
        # Define input rasters for the current dataset
        # input_rasters = [

        #     f'/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}//img1_cropped/',
        #     f'/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}//img2_cropped/'
        # ]

        input_rasters = [
            f'/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images//{dataset_name}/imgs_1/',
            f'/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images//{dataset_name}/imgs_2/'
        ]
        # Perform change detection for the current dataset
        change_mask = change_detection(dataset_name, input_rasters)

    except Exception as e:
        print(f"Error occurred in dataset: {dataset_name}")
        print(f"Error message: {str(e)}")





    print("Done")