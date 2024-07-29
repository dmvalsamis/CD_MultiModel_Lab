import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

# Hide the GPUs for timing the application
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp, conv_classifier_two_with_aspp
import matplotlib.pyplot as plt
from changedetection.utils.visualize import create_rgb
from numpy import expand_dims
import glob
from skimage import io
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import expand_dims as expand
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def plot_pred(y_pred_conv, img1, img2):
    unique_values = np.unique(y_pred_conv)
    print("Unique values in the predictions:", unique_values)

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

    plt.show()

def preprocess_image(image_path, size):
    img = load_img(image_path, target_size=size, color_mode='rgb')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def create_rgb(img):
    return (img - img.min()) / (img.max() - img.min())

def explain_model_with_lime(model, img1, img2, seg_method='quickshift'):
    def siamese_predict(images):
        img1_batch = np.repeat(np.expand_dims(img1, axis=0), len(images), axis=0)
        img2_batch = np.array(images)
        predictions = model.predict([img1_batch, img2_batch])

        # Assuming binary classification, return the mean probability of class '1'
        mean_predictions = predictions[:, :, :, 1].mean(axis=(1, 2))
        print(mean_predictions.shape)
        return mean_predictions  # Shape should be (num_samples,)

    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm(seg_method)
    
    # Convert to grayscale by averaging the color channels
    img2_gray = np.mean(img2, axis=2)
    
    # Normalize img2 for LIME
    img2_normalized = (img2_gray - img2_gray.mean()) / img2_gray.std()
    
    explanation = explainer.explain_instance(img2_normalized, siamese_predict, hide_color=0, num_samples=2, segmentation_fn=segmenter)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    
    plt.imshow(temp, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title('LIME Explanation')
    plt.axis('off')
    plt.show()

def process_image_pair(filename_a, filename_b):
    # Load images
    img1 = io.imread(filename_a)
    img2 = io.imread(filename_b)

    saved_model = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_Levir_8192.h5'         

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
    
    # Generate predictions
    prediction = cd_model.predict([img1_prepared, img2_prepared])
    y_pred_conv = np.argmax(prediction[0], axis=2)

    #plot_pred(y_pred_conv, img1, img2)

    # Apply LIME explanation
    explain_model_with_lime(cd_model, img1_normalized, img2_normalized)

# Define paths
input_path_a = '/data/valsamis_data/data/LEVIR-CD/original_set/total/A/'
input_path_b = '/data/valsamis_data/data/LEVIR-CD/original_set/total/B/'

# Process all image pairs
for file_a, file_b in zip(sorted(glob.glob(input_path_a + '*.png')), sorted(glob.glob(input_path_b + '*.png'))):
    process_image_pair(file_a, file_b)
    print(f"Processed and saved results for {os.path.basename(file_a)}")
