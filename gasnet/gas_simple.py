import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import uuid
import matplotlib.pyplot as plt
from gasnet.CDNet_L import CDNet_L
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
#from torch.utils.tensorboard import SummaryWriter
from utils.log_params import log_params_sim1

import os

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

onera_train_target = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_train_data/'
onera_test_target = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_test_data/'

train = pd.read_csv(onera_train_target + "dataset_train.csv")
test = pd.read_csv(onera_test_target + "dataset_test.csv")

train = train.sample(frac=1, random_state=1)
test = test.sample(frac=1, random_state=1)
print("Train Data", len(train))
print("Test Data", len(test))

n_ch = 3
channel = 'rgb'  

# Load training data
X_train1 = np.ndarray(shape=(len(train), 96, 96, n_ch))
X_train2 = np.ndarray(shape=(len(train), 96, 96, n_ch))
y_train = np.ndarray(shape=(len(train), 96, 96))

pos = 0
for index in train.index:
    img1 = np.load(onera_train_target + train['pair1'][index])
    img2 = np.load(onera_train_target + train['pair2'][index])
    X1 = create_rgb_onera(img1, channel)
    X2 = create_rgb_onera(img2, channel)
    X1 = (X1 - X1.mean()) / X1.std()
    X2 = (X2 - X2.mean()) / X2.std()
    X_train1[pos] = X1
    X_train2[pos] = X2
    y_train[pos] = np.load(onera_train_target + train['change_mask'][index])
    pos += 1

# Ensure target labels have the same shape as model output
y_train = np.expand_dims(y_train, axis=1)

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

# Create DataLoaders
train_data = TensorDataset(torch.tensor(X_train1).permute(0, 3, 1, 2).float(),
                           torch.tensor(X_train2).permute(0, 3, 1, 2).float(),
                           torch.tensor(y_train).float())
test_data = TensorDataset(torch.tensor(X_test1).permute(0, 3, 1, 2).float(),
                          torch.tensor(X_test2).permute(0, 3, 1, 2).float(),
                          torch.tensor(y_test).float())

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Define hook function
def print_layer_shape(module, input, output):
    print(f'{module.__class__.__name__}:')
    print(f'    Input shape: {input[0].shape}')
    print(f'    Output shape: {output[0].shape}')

# Register hooks
def register_hooks(model):
    for layer in model.children():
        layer.register_forward_hook(print_layer_shape)
        if len(list(layer.children())) > 0:
            register_hooks(layer)



# Model Training and Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CDNet_L().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



# Training loop
num_epochs = 30

model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/'

model_id = uuid.uuid4().hex[:4]
cd_model_name = "Gas_Net_"+"CBMI_"+model_id+".h5"

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs1, inputs2, labels) in enumerate(train_loader):
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    print(f"Logged Loss/train for epoch {epoch}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the model after training
save_path = os.path.join(model_path, cd_model_name)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Evaluate the model
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs1, inputs2, labels in test_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        outputs = model(inputs1, inputs2)
        predicted = (outputs > 0.5).float()
        
        all_labels.extend(labels.cpu().numpy().flatten())
        all_predictions.extend(predicted.cpu().numpy().flatten())

# Convert to numpy arrays for metric calculations
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = recall_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
specificity = tn / (tn + fp)


print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

log_params_sim1("Task 1", " ", ' ', " ", " ", "Softmax", " ", 'Adam', num_epochs, 'weighted_categorical_crossentropy', " ", recall, specificity, precision, f1, accuracy, "CBMI Set", 96, " ", "none", cd_model_name)

