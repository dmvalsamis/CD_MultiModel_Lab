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
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


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

cd_model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_Onera_0a8d.h5'
cd_model_name = 'CD_Simple_CBMI_0a8d'
model_id = cd_model_name.split('_')[-1].split('.')[0]


source_model.load_weights(cd_model_path)


branch_model = two_branch_cva_with_aspp(dropout, decay, depth, ImageSize, ImageSize, n_ch)
branch_model = two_feature_selector_cva_aspp(depth, source_model, branch_model)
#plot_model(branch_model, to_file='/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/graphs_cva/'+model_id+'_model_plot.png', show_shapes=True, show_layer_names=True)

    
# Data ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


onera_train_target =  '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_train_data/'  
onera_test_target = '/home/dvalsamis/Documents/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_test_data/'


onera_test_df = pd.read_csv(onera_test_target + "dataset_test.csv")

dsize = len(onera_test_df)  # Total size of the dataset
trsize = len(onera_train_target)  # Size of the training set
tesize = len(onera_test_target)  # Size of the testing set


X1 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))
X2 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))

nonNorm1 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))
nonNorm2 = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize,n_ch))

y = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))
y_pred = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))
y_pred_otsu = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))
y_pred_triangle = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))



for i in range(len(onera_test_df)):
    img1 =  np.load(onera_test_target+onera_test_df['pair1'][i])
    img2 = np.load(onera_test_target+onera_test_df['pair2'][i])
    img1 = create_rgb_onera(img1, channel)
    img2 = create_rgb_onera(img2, channel)

    nonNorm1[i] = img1
    nonNorm2[i] = img2

    if NORM:
        X1[i] = normalize(img1)
        X2[i] = normalize(img2)
    else:
        X1[i] = img1
        X2[i] = img2
    y[i] =  np.load(onera_test_target+onera_test_df['change_mask'][i])


branch_model.summary()

 
# Setting Predictions----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Final two Branch Model

for index in range(len(X1)):
    img1 = np.expand_dims(X1[index], axis=0)
    img2 = np.expand_dims(X2[index], axis=0)
    
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
    
    y_pred_otsu[index] = binary_otsu
    y_pred_triangle[index] = binary_triangle
    
y_pred_conv = source_model.predict([X1,X2])
y_pred_conv = np.argmax(y_pred_conv, axis=3)

print('Recall',recall(y,y_pred_conv))
print('Specificity',specificity(y,y_pred_conv))
print('Precision',precision(y,y_pred_conv)) 
print('F1',f_measure(y,y_pred_conv)) 
print('Accuracy',accuracy(y,y_pred_conv)) 





#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pos = random.randint(0, len(y))
print(pos)
print(len(y))
#pos = 157
#pos = 712

fig, ax = plt.subplots(3, 3, figsize=(10,10),constrained_layout=True)

ax[0,0].imshow(create_rgb(X1[pos], channel))
ax[0,0].set_title('Left Image', fontsize=20)
ax[0,0].axis('off')

ax[0,1].imshow(y[pos], cmap='gray')
ax[0,1].set_title('Ground Truth', fontsize=20)
ax[0,1].axis('off')

ax[0,2].imshow(create_rgb(X2[pos], channel))
ax[0,2].set_title('Right Image', fontsize=20)
ax[0,2].axis('off')

ax[1,0].imshow(create_rgb(X1[pos], channel))
ax[1,0].set_title('Left Image', fontsize=20)
ax[1,0].axis('off')

ax[1,1].imshow(y_pred_otsu[pos], cmap='gray')
ax[1,1].set_title('CVA(with Otsu)', fontsize=20)
ax[1,1].axis('off')

ax[1,2].imshow(create_rgb(X2[pos], channel))
ax[1,2].set_title('Right Image', fontsize=20)
ax[1,2].axis('off')

ax[2,0].imshow(create_rgb(X1[pos], channel))
ax[2,0].set_title('Left Image', fontsize=20)
ax[2,0].axis('off')

ax[2,1].imshow(y_pred_triangle[pos], cmap='gray')
ax[2,1].set_title('CVA(with Triagle)', fontsize=20)
ax[2,1].axis('off')

ax[2,2].imshow(create_rgb(X2[pos], channel))
ax[2,2].set_title('Right Image', fontsize=20)
ax[2,2].axis('off')



recall(y,y_pred_otsu)
get_roc(y,y_pred_otsu)


log_params_cva('CBMI_Test', model_id, cd_model_name, mtype, depth, dropout, decay, ImageSize, n_ch, channel, NORM)

#####Metrics#####
data_dict = {'Pretext':'Task 1',
             'Model ID': model_id,
             'Downstream':'CVA+Otsu(min size = 100)', 
             'Sensitivity/Recall':recall(y,y_pred_otsu), 
             'Specificity':specificity(y,y_pred_otsu), 
             'Precision':precision(y,y_pred_otsu), 
             'F1':f_measure(y,y_pred_otsu), 
             'Accuracy':accuracy(y,y_pred_otsu), 
             'Set':'CBMI Test', 
             'ImageSize':ImageSize,
             'Norm':NORM,
             'Pretext Model':'-', 
             'CD model':cd_model_name}

# Make data frame of above data
df = pd.DataFrame(data_dict, index=[0])

results_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/logs/cva_results_1.csv'

# append data frame to CSV file
#df.to_csv(results_path, mode='a', index=False, header=False)

# Append data to the CSV file
with open(results_path, 'a') as f:
    # If the file is empty, write the headers
    if f.tell() == 0:
        pd.DataFrame([data_dict.keys()]).to_csv(f, header=False, index=False)
    # Append data
    pd.DataFrame([data_dict.values()]).to_csv(f, header=False, index=False)

# print message
print("Data logged successfully.")

data_dict = {'Pretext':'Task 1',
             'Model ID': model_id,
             'Downstream':'CVA+Triangle(min size = 100)', 
             'Sensitivity/Recall':recall(y,y_pred_triangle), 
             'Specificity':specificity(y,y_pred_triangle), 
             'Precision':precision(y,y_pred_triangle), 
             'F1':f_measure(y,y_pred_triangle), 
             'Accuracy':accuracy(y,y_pred_triangle), 
             'Set':'CBMI Test', 
             'ImageSize':ImageSize,
             'Norm':NORM,
             'Pretext Model':'-', 
             'CD model':cd_model_name}

# Make data frame of above data
df = pd.DataFrame(data_dict, index=[0])
# append data frame to CSV file

# Append data to the CSV file
with open(results_path, 'a') as f:
    # If the file is empty, write the headers
    if f.tell() == 0:
        pd.DataFrame([data_dict.keys()]).to_csv(f, header=False, index=False)
    # Append data
    pd.DataFrame([data_dict.values()]).to_csv(f, header=False, index=False)
# print message
print("Data logged successfully.")


cd_results_df = pd.read_csv(results_path)

################fusion####################
from architectures.fusion_maria import fusion

y_pred_fusion = np.ndarray(shape=(len(onera_test_df),ImageSize,ImageSize))

for param in range(0, 13):
    for cm_pos in range(len(y_pred_fusion)):
        fused_mask = fusion(y_pred_triangle[cm_pos],y_pred_otsu[cm_pos],y_pred_conv[cm_pos], param)
        y_pred_fusion[cm_pos] = fused_mask
        
    data_dict = {'Pretext':'Task 1',
                 'Model ID': model_id,
                 'Downstream':'Fusion', 
                 'Sensitivity/Recall':recall(y,y_pred_fusion), 
                 'Specificity':specificity(y,y_pred_fusion), 
                 'Precision':precision(y,y_pred_fusion), 
                 'F1':f_measure(y,y_pred_fusion), 
                 'Accuracy':accuracy(y,y_pred_fusion), 
                 'Set':'CBMI Test', 
                 'ImageSize':ImageSize,
                 'Norm':"param="+str(param),
                 'Pretext Model':'-', 
                 'CD model':cd_model_name}
    
    # Make data frame of above data
    df = pd.DataFrame(data_dict, index=[0])
    # Append data to the CSV file
    with open(results_path, 'a') as f:
        # If the file is empty, write the headers
        if f.tell() == 0:
            pd.DataFrame([data_dict.keys()]).to_csv(f, header=False, index=False)
        # Append data
        pd.DataFrame([data_dict.values()]).to_csv(f, header=False, index=False)
    # print message
    print("Data logged successfully.", param)
    
#cd_results_df = pd.read_csv('cd_results.csv')

#plot fusion
cm_pos = 0
for cm_pos in range(len(y_pred_fusion)):
    fused_mask = fusion(y_pred_triangle[cm_pos],y_pred_otsu[cm_pos],y_pred_conv[cm_pos], 10)
    y_pred_fusion[cm_pos] = fused_mask

pos = random.randint(0, len(y))
print(pos)
print(len(y))
#pos = 157
#pos = 712
#pos = 577
fig, ax = plt.subplots(5, 1, figsize=(10,40),constrained_layout=True)

ax[0].imshow(y[pos], cmap='gray')
ax[0].set_title('Ground Truth', fontsize=30)
ax[0].axis('off')

ax[1].imshow(y_pred_conv[pos], cmap='gray')
ax[1].set_title('Conv classifier', fontsize=30)
ax[1].axis('off')

ax[2].imshow(y_pred_otsu[pos], cmap='gray')
ax[2].set_title('CVA(with Otsu)', fontsize=30)
ax[2].axis('off')

ax[3].imshow(y_pred_triangle[pos], cmap='gray')
ax[3].set_title('CVA(with Triagle)', fontsize=30)
ax[3].axis('off')

ax[4].imshow(y_pred_fusion[pos], cmap='gray')
ax[4].set_title('Fusion', fontsize=30)
ax[4].axis('off')

fig, ax = plt.subplots(1, 2, figsize=(10,40),constrained_layout=True)
ax[0].imshow(create_rgb(X1[pos], 'rgb'))
ax[0].set_title('Left Image', fontsize=20)
ax[0].axis('off')



ax[1].imshow(create_rgb(X2[pos], 'rgb'))
ax[1].set_title('Right Image', fontsize=20)
ax[1].axis('off')




print("End of first excecution")

