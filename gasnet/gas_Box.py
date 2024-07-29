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
import seaborn as sns
from gasnet.CDNet_L import CDNet_L
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
#from torch.utils.tensorboard import SummaryWriter
from utils.log_params import log_params_sim1

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define helper functions
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
channel = 'rgb'  # Set the channel according to your requirement

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


# Model Training and Evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCELoss()


num_epochs = 30
model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/'


# Initialize lists to store metrics
metrics_list = []

for run in range(64):
    model_id = uuid.uuid4().hex[:4]
    cd_model_name = f"Gas_Net_CBMI_{model_id}.h5"
    
    model = CDNet_L().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
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

    # Save the model after training
    save_path = os.path.join(model_path, cd_model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model {run + 1} saved to {save_path}")
    
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

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = recall_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    specificity = tn / (tn + fp)

    metrics_list.append({
        'Model': run + 1,
        'Recall': recall,
        'Specificity': specificity,
        'Precision': precision,
        'F1': f1,
        'Accuracy': accuracy
    })

    log_params_sim1("Task 1", " ", ' ', " ", " ", "Softmax", " ", 'Adam', num_epochs, 'weighted_categorical_crossentropy', " ", recall, specificity, precision, f1, accuracy, "CBMI Set", 96, " ", "none", cd_model_name)


    print(f"Run {run + 1}: Recall: {recall:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Convert metrics list to DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Save the metrics to a CSV file
metrics_df.to_csv('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/FCSiam/metrics_64_models.csv', index=False)

# Plotting the box plot for all metrics
plt.figure(figsize=(10, 6))
sns.boxplot(data=metrics_df.drop(columns=['Model']), palette="Set2")
plt.title('Boxplot of Model Metrics for 64 Runs')
plt.ylabel('Values')

# Save the plot
output_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/FCSiam/Box_plots/boxplot_metrics_64_runs.png'
plt.savefig(output_path)
print(f"Box plot saved to {output_path}")
