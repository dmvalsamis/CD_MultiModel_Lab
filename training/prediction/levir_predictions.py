import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')
import time

from architectures.similarity_detection import pretext_task_one_nopool,pretext_task_one_aspp

# Now you can use pretext_task_one_nopool in your script

#from tests import change_detection_noup, change_detection_noup_1x1convs
#from similarity_detection import pretext_task_one_nopool
import tensorflow
from tensorflow.keras import layers, Model
from architectures.branch import branches_triplet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import pandas as pd
import numpy as np
import os
from tensorflow import keras
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp,conv_classifier_two_with_aspp
from utils.layer_select import feature_selector, feature_selector_simple, transfer_learning_model #, feature_selector_task2
from utils.my_metrics import recall, accuracy, specificity, precision, f_measure, get_confusion_matrix
from utils.log_params import log_params_sim1

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import uuid
import random
from utils.weighted_cross_entropy import weighted_categorical_crossentropy
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob
from numpy import expand_dims

# Define paths
input_path_a = '/data/valsamis_data/data/LEVIR-CD/original_set/val/A/'
input_path_b = '/data/valsamis_data/data/LEVIR-CD/original_set/val/B/'
output_path = '/data/valsamis_data/data/LEVIR-CD/Predictions/Trainable_ASPP'
saved_model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_Levir_a43b.h5'



def process_image_pair(filename_a, filename_b):
    # Load images
    img1 = io.imread(filename_a)
    img2 = io.imread(filename_b)

    saved_model = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_Levir_8192.h5'         
    output_path = '/home/dvalsamis/Documents/data/LEVIR-CD/Predictions/Trainable_ASPP'


    # Check if the folder exists
    if not os.path.exists(output_path):
        # Create the folder
        os.makedirs(output_path)

    shape = img1.shape
    depth = 2
    dropout = 0.1
    decay = 0.0001
    ImageSize_X = shape[0]
    ImageSize_Y = shape[1]
    n_ch = shape[2]
    
    cd_model = conv_classifier_two_with_aspp(depth, dropout, decay, ImageSize_X, ImageSize_Y, n_ch)
    cd_model.load_weights(saved_model)

    # Normalize images
    img1_normalized = (img1 - img1.mean()) / img1.std()
    img2_normalized = (img2 - img2.mean()) / img2.std()

    # Expand dimensions for model input
    img1_prepared = expand_dims(img1_normalized, axis=0)
    img2_prepared = expand_dims(img2_normalized, axis=0)
    

    print("Preds :")
    # Generate predictions
    prediction = cd_model.predict([img1_prepared, img2_prepared])
    print(prediction.shape)
    print(prediction[0,0,0,0])
    print(prediction[0,0,0,1])

    y_pred_conv = np.argmax(prediction[0], axis=2)
    print(y_pred_conv[0])
    

    # Visualize and save results
    fig, ax = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
    ax[0].imshow(img1)
    ax[0].set_title('Time 1', fontsize=20)
    ax[0].axis('off')

    ax[1].imshow(img2)
    ax[1].set_title('Time 2', fontsize=20)
    ax[1].axis('off')

    ax[2].imshow(y_pred_conv, cmap='gray')
    ax[2].set_title('Conv Prediction', fontsize=20)
    ax[2].axis('off')

    base_name = os.path.splitext(os.path.basename(filename_a))[0]
    plt.savefig(os.path.join(output_path, f'{base_name}_conv_cm.png'))
    np.save(os.path.join(output_path, f'{base_name}_conv_cm.npy'), y_pred_conv)
    plt.close()

# Process all image pairs
for file_a, file_b in zip(sorted(glob.glob(input_path_a + '*.png')), sorted(glob.glob(input_path_b + '*.png'))):
    process_image_pair(file_a, file_b)
    print(f"Processed and saved results for {os.path.basename(file_a)}")
