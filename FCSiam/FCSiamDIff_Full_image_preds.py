import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import numpy as np
import torch
import os
import glob
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from FCSiam.fcsiaDiff import FCSiamDiff
from numpy import expand_dims

# Hide the GPUs for timing the application
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def read_rasters(path):
    im_names = glob.glob(os.path.join(path, '*B04.tif'))
    r = io.imread(im_names[0])
    im_names = glob.glob(os.path.join(path, '*B03.tif'))
    g = io.imread(im_names[0])
    im_names = glob.glob(os.path.join(path, '*B02.tif'))
    b = io.imread(im_names[0])
    I = np.stack((r, g, b), axis=2).astype('float')
    return I

def scaleMinMax(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val) * 255
    return x.astype(np.uint8)

def create_rgb(x):
    r = x[:, :, 2]
    r = scaleMinMax(r)
    g = x[:, :, 1]
    g = scaleMinMax(g)
    b = x[:, :, 0]
    b = scaleMinMax(b)
    rgb = np.dstack((r, g, b))
    return rgb

def normalize_image(img):
    return (img - img.mean()) / (img.std() + 1e-8)

def resize_image(img, target_shape):
    return resize(img, target_shape, mode='reflect', anti_aliasing=True)

# Adjust the shapes of the input images to be divisible by the downsampling factors
def adjust_image_shapes(img1, img2, factor=32):
    h, w = img1.shape[:2]
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    img1_resized = resize_image(img1, (new_h, new_w, img1.shape[2]))
    img2_resized = resize_image(img2, (new_h, new_w, img2.shape[2]))
    return img1_resized, img2_resized

def change_detection(dataset_name, input_rasters, model, device):
    # Load images
    imgs = []
    for timeframe in input_rasters:
        print(timeframe)
        imgs.append(read_rasters(timeframe))
    img1_or = imgs[0]
    img2_or = imgs[1]

    # Print original image shapes
    print(f"Original shape of img1_or: {img1_or.shape}")
    print(f"Original shape of img2_or: {img2_or.shape}")

    # Adjust the image shapes to be divisible by the downsampling factor
    img1_or, img2_or = adjust_image_shapes(img1_or, img2_or)

    # Normalize images for prediction
    img1 = normalize_image(img1_or)
    img2 = normalize_image(img2_or)
    
    # Expand dimensions for prediction
    img1 = expand_dims(img1, axis=0)
    img2 = expand_dims(img2, axis=0)

    # Stack images to create input tensor
    inputs1 = torch.tensor(img1).permute(0, 3, 1, 2).float()
    inputs2 = torch.tensor(img2).permute(0, 3, 1, 2).float()
    inputs = torch.stack((inputs1, inputs2), dim=1)

    # Move tensors to the device
    inputs = inputs.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(inputs)
        predicted = (outputs > 0.5).float().cpu().numpy().squeeze()

    # Save the prediction
    predictions_path = f'/data/valsamis_data/data/CBMI/CBMI_0.3/Predictions/Depth_2/SiamDiff/{dataset_name}'
    os.makedirs(predictions_path, exist_ok=True)
    pred_filename = os.path.join(predictions_path, 'prediction.npy')
    np.save(pred_filename, predicted)

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
    font = 20
    ax[0].imshow(create_rgb(img1_or))
    ax[0].set_title('t1', fontsize=font)
    ax[0].axis('off')
    ax[1].imshow(create_rgb(img2_or))
    ax[1].set_title('t2', fontsize=font)
    ax[1].axis('off')
    ax[2].imshow(predicted, cmap='gray')
    ax[2].set_title('FCSiamDiff Prediction', fontsize=font)
    ax[2].axis('off')
    plt.savefig(os.path.join(predictions_path, 'prediction.png'))
    plt.show()

# Read the dataset names from the all.txt file
dataset_names_file = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/all.txt'

with open(dataset_names_file, 'r') as file:
    dataset_names_line = file.readline().strip()  # Read the line containing all dataset names
    dataset_names = dataset_names_line.split(',')  # Split the line into individual dataset names

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = FCSiamDiff().to(device)

# Load the trained model
model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models_Torch/FCSiamDiff_CBMI_daa1.h5'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Iterate over dataset names
for dataset_name in dataset_names:
    try:
        # Define input rasters for the current dataset
        input_rasters = [
            f'/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}/img1_cropped/',
            f'/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI 0.3_initial/{dataset_name}/img2_cropped/'
        ]
        # Perform change detection for the current dataset
        change_detection(dataset_name, input_rasters, model, device)
    except Exception as e:
        print(f"Error occurred in dataset: {dataset_name}")
        print(f"Error message: {str(e)}")

print("Done")
