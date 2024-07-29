#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:40:56 2022

@author: aleoikon
"""

import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')
import time


from skimage.filters import threshold_otsu, threshold_triangle
from tensorflow import keras
from architectures.branch import branches_nopool, branch_cva, branch_cva_aspp,two_branch_cva_with_aspp,two_branch_cva_with_aspp_fmaps
#from tests import change_detection_noup, change_detection_noup_1x1convs
from architectures.similarity_detection import pretext_task_one_nopool
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import pandas as pd
import tensorflow as tf
import numpy as np
import os 
import random
from numpy import expand_dims
from skimage.morphology import remove_small_objects
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.layer_select import feature_selector_cva, feature_selector_cva_aspp,two_feature_selector_cva_aspp
from utils.log_params import log_params_cva
from architectures.conv_classifier import conv_classifier, conv_classifier_two,conv_classifier_two_with_aspp
from keras.models import Sequential
from skimage.filters import gaussian
from utils.my_metrics import recall, accuracy, specificity, precision, f_measure, get_roc
import uuid
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="1"


#Euclidean Distance 
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

def create_rgb_onera(x,channel):
    if channel == 'red':
        r = x[:,:,2]
        r = np.expand_dims(r, axis=2)
        return r
    if channel == 'green':
        g = x[:,:,1]
        g = np.expand_dims(g, axis=2)
        return g
    if channel == 'blue':
        b = x[:,:,0]
        b = np.expand_dims(b, axis=2)
        return b
    if channel == 'rgb':
        r = x[:,:,2]
        g = x[:,:,1]
        b  = x[:,:,0]
        rgb = np.dstack((r,g,b))
        return(rgb)
    if channel == 'rgbvnir':
        r = x[:,:,2]
        g = x[:,:,1]
        b  = x[:,:,0]
        vnir = x[:,:,3]
        rgbvnir = np.stack((r,g,b,vnir),axis=2).astype('float')
        return(rgbvnir)
    else:
        return x
        print("NOT CORRECT CHANNELS")

def normalize(x):
    img =((x - x.mean()) / x.std())
    return img

def scaleMinMax(x):
    return ((x - np.nanpercentile(x,2)) / (np.nanpercentile(x,98) - np.nanpercentile(x,2)))


def create_rgb(x, channels):
    if channels == 'red':
        r = x[:,:,0]
        r = scaleMinMax(r)
        return r
    if channels == 'green':    
        g = x[:,:,0]
        g = scaleMinMax(g)
        return g
    if channels == 'blue':
        b  = x[:,:,0]
        b = scaleMinMax(b)
        return b
    if channels == 'rgb':
        r = x[:,:,2]
        r = scaleMinMax(r)
        g = x[:,:,1]
        g = scaleMinMax(g)
        b  = x[:,:,0]
        b = scaleMinMax(b)
        rgb = np.dstack((r,g,b))
        return(rgb)
    


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

depth = 2
dropout = 0.1
decay = 0.0001
NORM = True
ImageSize = 96
n_ch = 3
channel = 'rgb'

# Models ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
source_model = conv_classifier_two_with_aspp(depth, dropout, decay, ImageSize, ImageSize, n_ch)
mtype = 'conv_classifier_two'

cd_model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_CBMI_c141.h5'
cd_model_name = 'CD_Simple_CBMI_c141'
model_id = cd_model_name.split('_')[-1].split('.')[0]


source_model.load_weights(cd_model_path)


branch_model,feature_model = two_branch_cva_with_aspp_fmaps(dropout, decay, depth, ImageSize, ImageSize, n_ch)
branch_model = two_feature_selector_cva_aspp(depth, source_model, branch_model)
#plot_model(branch_model, to_file='/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/graphs_cva/'+model_id+'_model_plot.png', show_shapes=True, show_layer_names=True)

    
# Data ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# onera_train_target =  '/home/aleoikon/Documents/data/ssl/onera_npys/patches/downstream/train/'  
# onera_test_target = '/home/aleoikon/Documents/data/ssl/onera_npys/patches/downstream/test/' 



onera_train_target =  '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_train_data/'  
onera_test_target = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_test_data/'


onera_test_df = pd.read_csv(onera_test_target + "dataset_test.csv")

dsize = len(onera_test_df)  # Total size of the dataset
trsize = len(onera_train_target)  # Size of the training set
tesize = len(onera_test_target)  # Size of the testing set


X1 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))
X2 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))
input_1 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))
input_2 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))
y = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))
y_pred_otsu = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))
y_pred_triangle = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))
y_pred_conv = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))



for i in range(len(onera_test_df)):
    img1 =  np.load(onera_test_target+onera_test_df['pair1'][i])
    img2 = np.load(onera_test_target+onera_test_df['pair2'][i])
    img1 = create_rgb_onera(img1, channel)
    img2 = create_rgb_onera(img2, channel)

    input_1[i] = img1
    input_2[i] = img2

    if NORM:
        X1[i] = normalize(img1)
        X2[i] = normalize(img2)
    else:
        X1[i] = img1
        X2[i] = img2
    y[i] =  np.load(onera_test_target+onera_test_df['change_mask'][i])


layers_of_interest = ['aspp_reduced_relu', 'abs_diff_2','reduced_aspp_output','output']

feature_maps = feature_model.predict([X1, X2])


branch_model.summary()


# y_pred_conv = branch_model.predict([X1,X2])
# y_pred_conv = np.argmax(y_pred_conv, axis=3)
# # Assuming y_pred is a list with one element per output layer
# y_pred_single = y[0]  # Assuming the first element is the desired output

# pos = 57
# fig, ax = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

# ax[0, 0].imshow(create_rgb(input_1[pos], channel))
# ax[0, 0].set_title('Left Image', fontsize=20)
# ax[0, 0].axis('off')

# ax[0, 1].imshow(y[pos], cmap='gray')
# ax[0, 1].set_title('Ground Truth', fontsize=20)
# ax[0, 1].axis('off')

# ax[0, 2].imshow(create_rgb(input_2[pos], channel))
# ax[0, 2].set_title('Right Image', fontsize=20)
# ax[0, 2].axis('off')

# ax[1, 0].imshow(create_rgb(input_1[pos], channel))
# ax[1, 0].set_title('Left Image', fontsize=20)
# ax[1, 0].axis('off')

# ax[1, 1].imshow(y[pos], cmap='gray')  # Adjust based on the actual structure
# ax[1, 1].set_title('Predicted Mask', fontsize=20)
# ax[1, 1].axis('off')

# ax[1, 2].imshow(create_rgb(input_2[pos], channel))
# ax[1, 2].set_title('Right Image', fontsize=20)
# ax[1, 2].axis('off')

# plt.show()

# print("")
 


# Feature Maps ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import matplotlib.pyplot as plt
# import numpy as np
# import os

# feature_maps_dir = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/fmaps'



# if not os.path.exists(feature_maps_dir):
#     os.makedirs(feature_maps_dir)

# # Function to plot a single feature map
# def plot_feature_map(feature_map, index, map_index, save_dir):
#     plt.imshow(feature_map, cmap='viridis')
#     plt.colorbar()
#     plt.title(f"Feature Map {map_index} for Sample {index}")
#     plt.savefig(os.path.join(save_dir, f"feature_map_{index}_{map_index}.png"))
#     plt.close()

# # Function to plot a composite image from all feature maps
# def plot_composite_feature_map(feature_maps, index, save_dir):
#     composite_map = np.mean(feature_maps, axis=-1)
#     plt.imshow(composite_map, cmap='viridis')
#     plt.colorbar()
#     plt.title(f"Composite Feature Map for Sample {index}")
#     plt.savefig(os.path.join(save_dir, f"composite_feature_map_{index}.png"))
#     plt.close()



# for index in range(10):
#     img1 = np.expand_dims(X1[index], axis=0)
#     img2 = np.expand_dims(X2[index], axis=0)
    
#     # Assuming branch_model.predict now returns ASPP feature maps and final output
#     aspp_feature_maps, _ = branch_model.predict([img1, img2])
#     feature_maps = aspp_feature_maps[0]  # Extract the first (and only) item in the batch

#     feature_maps = feature_model.predict([X1, X2])

#     layer_names = [layer.name for layer in source_model.layers if 'relu' in layer.name]
#     plot_feature_map(feature_maps, layer_names)

#     # Save the feature maps
#     feature_map_path = os.path.join(feature_maps_dir, f"feature_map_{index}.npy")
#     np.save(feature_map_path, feature_maps)



#     # # Optionally, plot each feature map individually
#     # for map_index in range(feature_maps.shape[-1]):  # Loop through each channel
#     #     plot_feature_map(feature_maps[:, :, map_index], index, map_index, feature_maps_dir)
    
#     # And/or plot a composite image
#     plot_composite_feature_map(feature_maps, index, feature_maps_dir)

# ----------------------------------------------------------------------------------------------------------------------

# def save_and_visualize_feature_maps(feature_maps, layer_names, directory, initial_inputs, example_index=0):
#     # Ensure the output directory exists
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     # Plot and save initial inputs for the specified example index
#     fig, axes = plt.subplots(1, len(initial_inputs), figsize=(10, 5))
#     for i, ax in enumerate(axes.flat):
#         input_image = initial_inputs[i][example_index]  # Select the example_index-th example
#         if input_image.shape[-1] == 1:  # If grayscale, reshape to remove the last dimension for plotting
#             input_image = input_image.reshape(input_image.shape[0], input_image.shape[1])
#         ax.imshow(input_image, cmap='viridis')
#         ax.set_title(f'Input {i+1}')
#         ax.axis('off')
#     plt.suptitle(f'Initial Inputs for Example {example_index}')
#     plt.savefig(os.path.join(directory, f'initial_input_{example_index}.png'))
#     plt.show()

#     # Process each feature map corresponding to each layer
#     for fmap, layer_name in zip(feature_maps, layer_names):
#         # Define the directory for this particular layer
#         layer_dir = os.path.join(directory, layer_name)
#         if not os.path.exists(layer_dir):
#             os.makedirs(layer_dir)

#         num_filters = fmap.shape[-1]
#         fig, axes = plt.subplots(1, min(num_filters, 20), figsize=(20, 2))

#         if num_filters == 1:
#             axes = [axes]  # Make it iterable

#         # Plot and save each filter's feature map for the specified example index
#         for i, ax in enumerate(axes):
#             if i < 20:
#                 feature_image = fmap[example_index, :, :, i]
#                 ax.imshow(feature_image, cmap='viridis')
#                 ax.set_title(f'Filter {i}')
#                 ax.axis('off')
#                 plot_filename = os.path.join(layer_dir, f'filter_{i}_example_{example_index}.png')
#                 plt.savefig(plot_filename)

#         plt.suptitle(f'Feature Maps from Layer: {layer_name} for Example {example_index}')
#         plt.show()

#         # Save each feature map as NPY file for the specified example index
#         for i in range(num_filters):
#             npy_filename = os.path.join(layer_dir, f'filter_{i}_example_{example_index}.npy')
#             np.save(npy_filename, fmap[example_index, :, :, i])

# # Example usage
# feature_maps_dir = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/fmaps_3'
# initial_inputs = [X1, X2]
# example_index = 0  # Change this to visualize different examples
# save_and_visualize_feature_maps(feature_maps, layers_of_interest, feature_maps_dir, initial_inputs, example_index)
# ----------------------------------------------------------------------------------------------------------------------


def save_and_visualize_feature_maps(feature_maps, layer_names, directory, initial_inputs, example_index=0, y_true=None, y_pred_conv=None):
    # Ensure the output directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Plot and save initial inputs and associated masks for the specified example index
    fig, axes = plt.subplots(1, len(initial_inputs) + 2, figsize=(15, 5))  # Adjusted for two additional plots
    for i, ax in enumerate(axes[:-2]):
        input_image = initial_inputs[i][example_index]
        if input_image.shape[-1] == 1:  # Handle grayscale images
            input_image = input_image.reshape(input_image.shape[0], input_image.shape[1])
        ax.imshow(input_image, cmap='viridis')
        ax.set_title(f'Input {i+1}')
        ax.axis('off')

    # Plot and save the ground truth and predicted mask
    if y_true is not None and y_pred_conv is not None:
        axes[-2].imshow(y_true[example_index], cmap='gray')
        axes[-2].set_title('Ground Truth')
        axes[-2].axis('off')
        axes[-1].imshow(y_pred_conv[example_index], cmap='gray')
        axes[-1].set_title('Predicted Mask')
        axes[-1].axis('off')

    plt.suptitle(f'Initial Inputs and Masks for Example {example_index}')
    plt.savefig(os.path.join(directory, f'input_and_masks_{example_index}.png'))
    plt.show()

    # Process and save each feature map corresponding to each layer
    for fmap, layer_name in zip(feature_maps, layer_names):
        layer_dir = os.path.join(directory, layer_name)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)

        num_filters = fmap.shape[-1]
        cols = 8  # Define the number of columns
        rows = min(4, (num_filters + cols - 1) // cols)  # Define rows, maximum of 4 rows
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = axes.flatten()  # Flatten the array for easier indexing

        # Plot each feature image in its respective subplot
        for i in range(min(num_filters, 32)):  # Show up to 32 filters
            if i < len(axes):  # Check if there are enough subplots available
                ax = axes[i]
                feature_image = fmap[example_index, :, :, i]
                ax.imshow(feature_image, cmap='viridis')
                ax.set_title(f'Filter {i}', fontsize=10)
                ax.axis('off')
                plot_filename = os.path.join(layer_dir, f'filter_{i}_example_{example_index}.png')
                plt.savefig(plot_filename)  # Save the image file for this filter

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.suptitle(f'Feature Maps from Layer: {layer_name} for Example {example_index}', fontsize=16)
        plt.show()

        # Save each feature map as NPY file
        for i in range(num_filters):
            npy_filename = os.path.join(layer_dir, f'filter_{i}_example_{example_index}.npy')
            np.save(npy_filename, fmap[example_index, :, :, i])

# Example usage
feature_maps_dir = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/fmaps/fmaps_1'
initial_inputs = [X1, X2]
example_index = 57  # Change this to visualize different examples
y_true = y
y_pred_conv = source_model.predict([X1, X2])
y_pred_conv = np.argmax(y_pred_conv, axis=3)
save_and_visualize_feature_maps(feature_maps, layers_of_interest, feature_maps_dir, initial_inputs, example_index, y_true, y_pred_conv)


print("Done")
