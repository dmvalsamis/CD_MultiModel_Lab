import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from skimage import io
from torch.utils.data import DataLoader, TensorDataset

from FCSiam.fcsiamConc import FCSiamConc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def normalize_image(img):
    return (img - img.mean()) / img.std()

def process_image_pair(filename_a, filename_b, model, device, output_path):
    # Load images
    img1 = io.imread(filename_a)
    img2 = io.imread(filename_b)

    # Normalize images
    img1_normalized = normalize_image(img1)
    img2_normalized = normalize_image(img2)

    # Convert images to PyTorch tensors and reshape to fit model input
    img1_tensor = torch.tensor(img1_normalized).permute(2, 0, 1).unsqueeze(0).float()  # [1, channels, height, width]
    img2_tensor = torch.tensor(img2_normalized).permute(2, 0, 1).unsqueeze(0).float()  # [1, channels, height, width]

    inputs = torch.stack((img1_tensor, img2_tensor), dim=1)  # [1, 2, channels, height, width]

    # Model inference
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted_mask = (outputs > 0.5).float().cpu().numpy()[0, 0]

    # Visualize and save results
    fig, ax = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
    ax[0].imshow(img1)
    ax[0].set_title('Time 1', fontsize=20)
    ax[0].axis('off')

    ax[1].imshow(img2)
    ax[1].set_title('Time 2', fontsize=20)
    ax[1].axis('off')

    ax[2].imshow(predicted_mask, cmap='gray')
    ax[2].set_title('Predicted Change Mask', fontsize=20)
    ax[2].axis('off')

    base_name = os.path.splitext(os.path.basename(filename_a))[0]
    plt.savefig(os.path.join(output_path, f'{base_name}_predicted_mask.png'))
    np.save(os.path.join(output_path, f'{base_name}_predicted_mask.npy'), predicted_mask)
    plt.close()

# Define paths
input_path_a = '/data/valsamis_data/data/LEVIR-CD/original_set/val/A/'
input_path_b = '/data/valsamis_data/data/LEVIR-CD/original_set/val/B/'
output_path = '/data/valsamis_data/data/LEVIR-CD/Predictions/Trainable_ASPP'
model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/FCSiamConc_CBMI_ac4f.h5'

# Ensure output path exists
os.makedirs(output_path, exist_ok=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCSiamConc().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Process all image pairs
for file_a, file_b in zip(sorted(glob.glob(input_path_a + '*.png')), sorted(glob.glob(input_path_b + '*.png'))):
    process_image_pair(file_a, file_b, model, device, output_path)
    print(f"Processed and saved results for {os.path.basename(file_a)}")
