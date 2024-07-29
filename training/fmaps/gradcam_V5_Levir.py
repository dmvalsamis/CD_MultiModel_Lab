import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import numpy as np
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp, conv_classifier_two_with_aspp
from architectures.branch import branch_cva, branch_cva_with_nspp, two_branch_cva_with_aspp, two_branch_cva_with_aspp_fmaps
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
from tensorflow import keras
import cv2

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
from numpy import expand_dims
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from skimage import io

# Define paths
input_path_a = '/home/dvalsamis/Documents/data/LEVIR-CD/original_set/val/A/'
input_path_b = '/home/dvalsamis/Documents/data/LEVIR-CD/original_set/val/B/'

def sigmoid(x, a, b, c):

    return c / (1 + np.exp(-a * (x - b)))


def grad_cam(model, img1_array, img2_array, layer_name, class_idx, eps=1e-8):

    # Create a model that maps the input images to the activations of the specified layer as well as the output predictions
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])


    # Find all Conv2D layers
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

    
    if len(conv_layers) < 2:
        raise ValueError("The model does not have at least two Conv2D layers.")

    # Get the second-to-last Conv2D layer
    second_last_conv_layer = conv_layers[-2]
    target_layer = model.get_layer(second_last_conv_layer.name)
    print(second_last_conv_layer.name)


    with tf.GradientTape() as tape:

        # Cast the images to float32 and watch them
        inputs = [tf.cast(img1_array, tf.float32), tf.cast(img2_array, tf.float32)]
        tape.watch(inputs)
        # Pass the images through the model to get the output of the specified layer and the final output
        (convOutputs, predictions) = grad_model(inputs)
        unique_values = np.unique(predictions)
        print("Unique values in the grads:", unique_values)
        # Focus on the loss associated with the specific class index
        loss = predictions[:, class_idx]

    # Compute the gradients of the loss with respect to the outputs of the specified layer
    grads = tape.gradient(loss, convOutputs)
    print("Gradients max, min:", np.max(grads), np.min(grads)) 

    # Compute guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads

    # Remove the batch dimension
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    # Compute the average of the gradient values, and use them as weights to compute the weighted sum of the filters
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    
    cam = tf.maximum(cam, 0)
    print("CAM max, min:", np.max(cam), np.min(cam))  # Debug CAM values

    # Resize the CAM to the size of the input image and normalize
    w, h = img2_array.shape[1], img2_array.shape[2]
    heatmap = cv2.resize(cam.numpy(), (w, h))
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    return heatmap

def preprocess_image(image_path, size):
    img = load_img(image_path, target_size=size, color_mode='rgb')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

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

    plot_pred(y_pred_conv, img1, img2)


    img_array_a = preprocess_image(filename_a, (1024, 1024))
    img_array_b = preprocess_image(filename_b, (1024, 1024))

    layer_name = 'reduced_aspp_output'
    class_idx = 1
    
    heatmap = grad_cam(cd_model, img_array_a, img_array_b, layer_name, class_idx)
    display_heatmap(heatmap)

    img_bgr = io.imread(filename_b)  # Assuming the original image is in BGR format
    superimposed_image = superimpose(img_bgr, heatmap, thresh=0.5, emphasize=True)

    plt.imshow(superimposed_image)
    plt.axis('off')
    plt.show()



# Process all image pairs
for file_a, file_b in zip(sorted(glob.glob(input_path_a + '*.png')), sorted(glob.glob(input_path_b + '*.png'))):
    process_image_pair(file_a, file_b)
    print(f"Processed and saved results for {os.path.basename(file_a)}")
