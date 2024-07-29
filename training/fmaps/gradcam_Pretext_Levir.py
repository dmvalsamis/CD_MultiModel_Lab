import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import numpy as np
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp, conv_classifier_two_with_aspp
from numpy import expand_dims
import glob
import os
from skimage import io
from tensorflow import keras
import cv2
from architectures.similarity_detection import pretext_task_one_nopool, pretext_one, pretext_task_one_aspp


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="1"


import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage import io

# Define paths
input_path_a = '/data/valsamis_data/data/LEVIR-CD/original_set/total/A/'
input_path_b = '/data/valsamis_data/data/LEVIR-CD/original_set/total/B/'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations of the last conv layer as well as the outputs
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen) with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by 'how important this channel is' with respect to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap

def apply_gradcam(img1, img2, model, last_conv_layer_name='joint', IMG_SIZE=(96, 96)):
    # Preprocessing the images
    img1 = tf.image.resize(img1, IMG_SIZE)
    img2 = tf.image.resize(img2, IMG_SIZE)
    img_array = [img1, img2]

    # Generate heatmap with the model predictions
    heatmap1 = make_gradcam_heatmap(img_array[0][tf.newaxis, ...], model, last_conv_layer_name, None)
    heatmap2 = make_gradcam_heatmap(img_array[1][tf.newaxis, ...], model, last_conv_layer_name, None)

    # Display heatmaps
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1.numpy().astype("uint8"))
    plt.imshow(heatmap1.numpy(), cmap='viridis', alpha=0.6)
    plt.title('Image 1 Grad-CAM')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2.numpy().astype("uint8"))
    plt.imshow(heatmap2.numpy(), cmap='viridis', alpha=0.6)
    plt.title('Image 2 Grad-CAM')
    plt.axis('off')

    plt.show()




def preprocess_image(image_path, size):
    img = load_img(image_path, target_size=size, color_mode='rgb')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


def plot_pred(y_pred_conv, img1, img2) :

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

def display_heatmap(heatmap):
    heatmap = np.maximum(heatmap, 0)  # ReLU to discard negative values
    heatmap /= np.max(heatmap)  # Normalize to [0,1] to improve visibility
    plt.imshow(heatmap, cmap='viridis')  # Use a colormap that enhances small differences
    plt.colorbar()
    plt.show()

def process_image_pair(filename_a, filename_b):
    # Load images
    img1 = io.imread(filename_a)
    img2 = io.imread(filename_b)

    saved_model = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/model_pretext1_unclouded_results.h5'         


    shape = img1.shape
    depth = 2
    dropout = 0.1
    decay = 0.0001
    ImageSize_X = shape[0]
    ImageSize_Y = shape[1]
    n_ch = shape[2]
    
    cd_model = pretext_task_one_nopool(dropout, decay, ImageSize_X, ImageSize_Y, n_ch)
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

    plot_pred(y_pred_conv, img1, img2)


    img_array_a = preprocess_image(filename_a, (1024, 1024))
    img_array_b = preprocess_image(filename_b, (1024, 1024))

    apply_gradcam(img1_prepared, img2_prepared, cd_model, last_conv_layer_name='joint')




# Process all image pairs
for file_a, file_b in zip(sorted(glob.glob(input_path_a + '*.png')), sorted(glob.glob(input_path_b + '*.png'))):
    process_image_pair(file_a, file_b)
    print(f"Processed and saved results for {os.path.basename(file_a)}")
