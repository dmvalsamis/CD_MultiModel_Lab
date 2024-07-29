import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
from gasnet.CDNet_L import CDNet_L
import matplotlib.pyplot as plt



os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def create_rgb_onera(x, channel):
    if channel == 'red':
        r = x[:, :, 2]
        r = np.expand_dims(r, axis=2)
        return r
    if channel == 'green':
        g = x[:, :, 1]
        g = np.expand_dims(g, axis=2)
        return g
    if channel == 'blue':
        b = x[:, :, 0]
        b = np.expand_dims(b, axis=2)
        return b
    if channel == 'rgb':
        r = x[:, :, 2]
        g = x[:, :, 1]
        b = x[:, :, 0]
        rgb = np.dstack((r, g, b))
        return rgb
    if channel == 'rgbvnir':
        r = x[:, :, 2]
        g = x[:, :, 1]
        b = x[:, :, 0]
        vnir = x[:, :, 3]
        rgbvnir = np.stack((r, g, b, vnir), axis=2).astype('float')
        return rgbvnir
    else:
        print("NOT CORRECT CHANNELS")
        return x

# Data Loading and Preparation

onera_test_target = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_test_data/'

test = pd.read_csv(onera_test_target + "dataset_test.csv")
test = test.sample(frac=1, random_state=1).head(20)  
print("Test Data", len(test))

n_ch = 3
channel = 'rgb'  

# Load test data
X_test1 = np.ndarray(shape=(len(test), 96, 96, n_ch))
X_test2 = np.ndarray(shape=(len(test), 96, 96, n_ch))
y_test = np.ndarray(shape=(len(test), 96, 96))

pos = 0
for index in test.index:
    img1 = np.load(onera_test_target + test['pair1'][index])
    img2 = np.load(onera_test_target + test['pair2'][index])
    X1 = create_rgb_onera(img1, channel)
    X2 = create_rgb_onera(img2, channel)
    X1 = (X1 - X1.mean()) / X1.std()
    X2 = (X2 - X2.mean()) / X2.std()
    X_test1[pos] = X1
    X_test2[pos] = X2
    y_test[pos] = np.load(onera_test_target + test['change_mask'][index])
    pos += 1

# Ensure target labels have the same shape as model output
y_test = np.expand_dims(y_test, axis=1)

# Create DataLoader
test_data = TensorDataset(torch.tensor(X_test1).permute(0, 3, 1, 2).float(),
                          torch.tensor(X_test2).permute(0, 3, 1, 2).float(),
                          torch.tensor(y_test).float())
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Model Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDNet_L().to(device)

# Load the trained model
model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/FCSiamDiff_CBMI_6acc.h5'
model.load_state_dict(torch.load(model_path))
model.eval()

# Directory to save predictions
predictions_path = '/data/valsamis_data/data/CBMI/CBMI_0.3/Predictions/Depth_2/fcsiamDiff'
os.makedirs(predictions_path, exist_ok=True)

all_predictions = []

with torch.no_grad():
    for i, (inputs1, inputs2, true_labels) in enumerate(test_loader):
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        outputs = model(inputs1, inputs2)
        predicted = (outputs > 0.5).float().cpu().numpy()

        for j in range(predicted.shape[0]):
            pred_filename = os.path.join(predictions_path, f'prediction_{i*test_loader.batch_size + j}.npy')
            np.save(pred_filename, predicted[j])
            all_predictions.append((inputs1[j].cpu().numpy(), inputs2[j].cpu().numpy(), predicted[j]))

            # Visualization and saving as .png
            plt.figure(figsize=(20, 5))

            plt.subplot(1, 4, 1)
            plt.imshow(inputs1[j].cpu().permute(1, 2, 0).numpy())
            plt.title('Input Image 1')

            plt.subplot(1, 4, 2)
            plt.imshow(inputs2[j].cpu().permute(1, 2, 0).numpy())
            plt.title('Input Image 2')

            plt.subplot(1, 4, 3)
            plt.imshow(predicted[j][0], cmap='gray')
            plt.title('Predicted Change Mask')

            plt.subplot(1, 4, 4)
            plt.imshow(true_labels[j][0].cpu().numpy(), cmap='gray')
            plt.title('True Change Mask')

            png_filename = os.path.join(predictions_path, f'prediction_{i*test_loader.batch_size + j}.png')
            plt.savefig(png_filename)
            plt.close()

print(f"Predictions saved to {predictions_path}")
