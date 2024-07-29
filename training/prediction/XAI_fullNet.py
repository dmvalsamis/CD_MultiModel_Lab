import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

#hide the gpus for timing the application
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp, conv_classifier_two_with_aspp
import matplotlib.pyplot as plt
from numpy import expand_dims

import glob
from skimage import io
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob
from numpy import expand_dims
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage import io

def extract_subnetworks(cd_model, depth):
    # Extract layers for the first branch
    input_1 = cd_model.input[0]
    output_1 = None
    
    for i, layer in enumerate(cd_model.layers):
        if layer.name == f'dropout{depth}_1':
            output_1 = layer.output
            break
    
    subnetwork1 = Model(inputs=input_1, outputs=output_1, name='subnetwork1')
    
    # Extract layers for the second branch
    input_2 = cd_model.input[1]
    output_2 = None
    
    for i, layer in enumerate(cd_model.layers):
        if layer.name == f'dropout{depth}_2':
            output_2 = layer.output
            break
    
    subnetwork2 = Model(inputs=input_2, outputs=output_2, name='subnetwork2')
    
    return subnetwork1, subnetwork2

def get_grad_cam_subnetwork(subnetwork, img, layer_name):

    grad_model = Model(inputs=subnetwork.inputs, outputs=[subnetwork.get_layer(layer_name).output, subnetwork.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (img.shape[2], img.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    return heatmap

def sigmoid(x, a, b, c):

    return c / (1 + np.exp(-a * (x - b)))

def superimpose(img_bgr, cam, thresh, emphasize=False):
    
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb

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

def preprocess_image(image_path, size):
    img = load_img(image_path, target_size=size, color_mode='rgb')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img


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
   
    subnetwork1, subnetwork2 = extract_subnetworks(cd_model, depth)
    subnetwork1.summary()
    subnetwork2.summary()

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


    # Example usage for Grad-CAM
    layer_name_1 = 'conv2_1' 
    layer_name_2 = 'conv2_2'
    img1 = preprocess_image(filename_a, (1024, 1024))
    img2 = preprocess_image(filename_b, (1024, 1024))

    heatmap1 = get_grad_cam_subnetwork(subnetwork1, img1, layer_name_1)
    heatmap2 = get_grad_cam_subnetwork(subnetwork2, img2, layer_name_2)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(img1[0])
    # plt.imshow(heatmap1, cmap='jet', alpha=0.5)
    # plt.title('Grad-CAM for img1')

    # plt.subplot(1, 2, 2)
    # plt.imshow(img2[0])
    # plt.imshow(heatmap2, cmap='jet', alpha=0.5)
    # plt.title('Grad-CAM for img2')

    # plt.show()

    img_bgr_1 = io.imread(filename_a)  # Assuming the original image is in BGR format
    img_bgr_2 = io.imread(filename_b)

    # Extract file name without extension
    base_name = os.path.basename(filename_a).split('.')[0]

    # Locate the label file
    label_path = os.path.join('/data/valsamis_data/data/LEVIR-CD/original_set/labels', base_name + '.png')
    label_img = io.imread(label_path)

    # Superimpose heatmaps on the original images
    superimposed_image1 = superimpose(img_bgr_1, heatmap1, thresh=0.5, emphasize=True)
    superimposed_image2 = superimpose(img_bgr_2, heatmap2, thresh=0.5, emphasize=True)

    # Plot both superimposed images and the label image in one figure
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(superimposed_image1, cv2.COLOR_BGR2RGB))
    plt.title('Superimposed Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(superimposed_image2, cv2.COLOR_BGR2RGB))
    plt.title('Superimposed Image 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(label_img, cmap='gray')
    plt.title('Label Image')
    plt.axis('off')

    plt.show()


input_path_a = '/data/valsamis_data/data/LEVIR-CD/original_set/total/A/'
input_path_b = '/data/valsamis_data/data/LEVIR-CD/original_set/total/B/'

# Process all image pairs
for file_a, file_b in zip(sorted(glob.glob(input_path_a + '*.png')), sorted(glob.glob(input_path_b + '*.png'))):
    process_image_pair(file_a, file_b)
    print(f"Processed and saved results for {os.path.basename(file_a)}")