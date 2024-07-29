import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

#hide the gpus for timing the application
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp, conv_classifier_two_with_aspp
import matplotlib.pyplot as plt
from changedetection.utils.visualize import create_rgb
from numpy import expand_dims

import glob
from skimage import io






def read_rasters(path):
    im_names = glob.glob(os.path.join(path, '*B04.tif'))  # search for files with names containing 'B04.tif'
    r = io.imread(im_names[0])
    im_names = glob.glob(os.path.join(path, '*B03.tif'))  # search for files with names containing 'B03.tif'
    g = io.imread(im_names[0])
    im_names = glob.glob(os.path.join(path, '*B02.tif'))  # search for files with names containing 'B02.tif'
    b = io.imread(im_names[0])
    I = np.stack((r, g, b), axis=2).astype('float')
    return I

def change_detection(dataset_name, input_rasters):
    # Load images
    imgs = []
    for timeframe in input_rasters:
        imgs.append(read_rasters(timeframe))
    img1_or = imgs[0]
    img2_or = imgs[1]

    # Load change detection model
    path = f'/home/dvalsamis/Documents/data/Onera/Predictions/Depth_2/Trainable_Onera/{dataset_name}'
    saved_model = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_CBMI_1ca4.h5'

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

    # Normalize images
    img1 = expand_dims(img1_or, axis=0)
    img1 = (img1 - img1.mean()) / img1.std()
    img2 = expand_dims(img2_or, axis=0)
    img2 = (img2 - img2.mean()) / img2.std()

    # Conv predictions
    prediction = cd_model.predict([img1, img2])
    y_preds = np.argmax(prediction[0], axis=2)
    y_pred_conv = y_preds

    # Display the original images and predictions
    fig, ax = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
    font = 20
    ax[0].imshow(create_rgb(img1_or))
    ax[0].set_title('t1', fontsize=font)
    ax[0].axis('off')
    ax[1].imshow(create_rgb(img2_or))
    ax[1].set_title('t2', fontsize=font)
    ax[1].axis('off')
    ax[2].imshow(y_pred_conv, cmap='gray')
    ax[2].set_title('Conv Prediction', fontsize=font)
    ax[2].axis('off')

    plt.show()



# Read the dataset names from the all.txt file
dataset_names_file = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/all.txt'

with open(dataset_names_file, 'r') as file:
    dataset_names_line = file.readline().strip()  # Read the line containing all dataset names
    dataset_names = dataset_names_line.split(',')  # Split the line into individual dataset names

# Iterate over dataset names
for dataset_name in dataset_names:
    try:
        # Define input rasters for the current dataset
        input_rasters = [
            f'/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}//img1_cropped/',
            f'/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}//img2_cropped/'
        ]

        # Perform change detection for the current dataset
        change_mask = change_detection(dataset_name, input_rasters)

    except Exception as e:
        print(f"Error occurred in dataset: {dataset_name}")
        print(f"Error message: {str(e)}")
