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
from architectures.branch import branch_cva, branch_cva_with_nspp, two_branch_cva_with_aspp,two_branch_cva_with_aspp_fmaps
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

def save_and_visualize_feature_maps(feature_maps, feature_model, source_model, layers_of_interest, img1, img2, feature_maps_dir):
    # Predict feature maps and classification/prediction masks
    feature_maps = feature_model.predict([img1, img2])
    y_pred_conv = source_model.predict([img1, img2])
    y_pred_conv = np.argmax(y_pred_conv, axis=3)  # Assuming a classification task

    # Directory handling
    if not os.path.exists(feature_maps_dir):
        os.makedirs(feature_maps_dir)

    # Setting up the plot for initial inputs and prediction results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Added more plots to visualize all necessary components

    # Showing the first image in the batch
    example_index = 0  # assuming batch size or single image

    # Displaying the original images
    axes[0].imshow(img1[example_index])  # Assuming img1 is properly preprocessed if necessary
    axes[0].set_title('Input Image 1')
    axes[0].axis('off')

    axes[1].imshow(img2[example_index])  # Assuming img2 is properly preprocessed if necessary
    axes[1].set_title('Input Image 2')
    axes[1].axis('off')

    # Displaying the predicted mask
    axes[2].imshow(y_pred_conv[example_index], cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    # Saving the initial inputs visualization
    plt.suptitle('Initial Inputs and Predicted Mask')
    plt.savefig(os.path.join(feature_maps_dir, 'inputs_and_predictions.png'))
    plt.show()

    # Visualize and save feature maps
    for fmap, layer_name in zip(feature_maps, layers_of_interest):
        layer_dir = os.path.join(feature_maps_dir, layer_name)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)

        num_filters = fmap.shape[-1]
        cols = 8  # Define columns
        rows = min(4, (num_filters + cols - 1) // cols)  # Define rows
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = axes.flatten()

        # Plot each feature map
        for i in range(min(num_filters, 32)):
            if i < len(axes):
                ax = axes[i]
                feature_image = fmap[example_index, :, :, i]
                ax.imshow(feature_image, cmap='viridis')
                ax.set_title(f'Filter {i}', fontsize=10)
                ax.axis('off')
                plt.savefig(os.path.join(layer_dir, f'filter_{i}_example_{example_index}.png'))

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.suptitle(f'Feature Maps from Layer: {layer_name}', fontsize=16)
        plt.show()

        # Save each feature map as an NPY file
        for i in range(num_filters):
            npy_filename = os.path.join(layer_dir, f'filter_{i}_example_{example_index}.npy')
            np.save(npy_filename, fmap[example_index, :, :, i])


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
    
    source_model = conv_classifier_two_with_aspp(depth, dropout, decay, ImageSize_X, ImageSize_Y, n_ch)

    cd_model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_CBMI_c141.h5'
    source_model.load_weights(cd_model_path)


    branch_model,feature_model = two_branch_cva_with_aspp_fmaps(dropout, decay, depth, ImageSize_X, ImageSize_Y, n_ch)
    branch_model = two_feature_selector_cva_aspp(depth, source_model, branch_model)
    
    #predictions
    img1 = expand_dims(img1_or, axis=0)
    img1 = (img1 - img1.mean()) / img1.std()
    img2 = expand_dims(img2_or, axis=0)
    img2 = (img2 - img2.mean()) / img2.std()
   

    layers_of_interest = ['aspp_reduced_relu', 'abs_diff_2','reduced_aspp_output','output']

    

    # Example usage
    feature_maps_dir = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/fmaps/fmaps_total_0'
    layers_of_interest = ['aspp_reduced_relu', 'abs_diff_2','reduced_aspp_output','output']

    save_and_visualize_feature_maps(None, feature_model, source_model, layers_of_interest, img1, img2, feature_maps_dir)


# Read the dataset names from the all.txt file

dataset_names_file = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI 0.3_initial/all.txt'

#dataset_names_file =  '/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images/all.txt'


with open(dataset_names_file, 'r') as file:
    dataset_names_line = file.readline().strip()  # Read the line containing all dataset names
    dataset_names = dataset_names_line.split(',')  # Split the line into individual dataset names

# Iterate over dataset names
for dataset_name in dataset_names:
    try:
        # Define input rasters for the current dataset
        input_rasters = [

            f'/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}//img1_cropped/',
            f'/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}//img2_cropped/'
        ]

        # input_rasters = [
        #     f'/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images//{dataset_name}/imgs_1/',
        #     f'/home/aleoikon/Documents/data/onera/Onera Satellite Change Detection dataset - Images//{dataset_name}/imgs_2/'
        # ]
        # Perform change detection for the current dataset
        change_mask = change_detection(dataset_name, input_rasters)

    except Exception as e:
        print(f"Error occurred in dataset: {dataset_name}")
        print(f"Error message: {str(e)}")




    print("Done")