#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:15:48 2022

@author: aleoikon
"""

import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')
import time

from architectures.similarity_detection import pretext_task_one_nopool, pretext_task_one_aspp
import tensorflow
from tensorflow.keras import layers, Model
from architectures.branch import branches_triplet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


import pandas as pd
import numpy as np
import os
from tensorflow import keras
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp, conv_classifier_two_with_aspp
from utils.layer_select import feature_selector_simple
from utils.my_metrics import recall, accuracy, specificity, precision, f_measure, get_confusion_matrix
from utils.log_params import log_params_sim1

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import random
from utils.weighted_cross_entropy import weighted_categorical_crossentropy
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

channel = 'rgb'

# Set random seed for TensorFlow and NumPy
tensorflow.random.set_seed(1234)
np.random.seed(1234)

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

def generate_short_id():
    unique_id = uuid.uuid4()
    short_id = str(unique_id.hex)[:4]
    return short_id

def scaleMinMax(x):
    return ((x - np.nanpercentile(x, 2)) / (np.nanpercentile(x, 98) - np.nanpercentile(x, 2)))

def create_rgb(x, channel):
    if channel == 'rgb':
        r = x[:, :, 2]
        r = scaleMinMax(r)
        g = x[:, :, 1]
        g = scaleMinMax(g)
        b = x[:, :, 0]
        b = scaleMinMax(b)
        rgb = np.dstack((r, g, b))
        return rgb
    else:
        one = x[:, :, 0]
        one = scaleMinMax(one)
        return one        

onera_train_target =  '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_train_data/'  
onera_test_target = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_test_data/'

train = pd.read_csv(onera_train_target + "dataset_train.csv")
test = pd.read_csv(onera_test_target + "dataset_test.csv")

train = train.sample(frac=1, random_state=1)
test = test.sample(frac=1, random_state=1)
print("Train Data", len(train))
print("Test Data", len(test))

NORM = True
n_ch = 3
EPOCHS = 55
BATCH_SIZE = 5

# Initialize a list to store the metrics
metrics_list = []

# Run the training loop 64 times
for run in range(64):
    print(f"Run {run+1}/64")
    
    # Load everything in memory
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

    y_hot_train = keras.utils.to_categorical(y_train, num_classes=2)
    
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

    y_hot_test = keras.utils.to_categorical(y_test, num_classes=2)

    # Create and compile model
    depth = 2
    dropout = 0.1
    decay = 0.0001
    LEARNING_RATE = 0.001

    cd_model = conv_classifier_two(depth, dropout, decay, 96, 96, n_ch)
    sim_model = pretext_task_one_nopool(dropout, decay, 96, 96, n_ch)

    pretext_model_name = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/model_pretext1_unclouded_results.h5'
    sim_model.load_weights(pretext_model_name)

    cd_model = feature_selector_simple(depth, sim_model, cd_model)

    weights = np.array([0.1, 0.2])
    optimizer = Adam(learning_rate=LEARNING_RATE)
    cd_model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])

    # Train the model
    history = cd_model.fit(
        [X_train1, X_train2],
        y_hot_train,
        validation_data=([X_test1, X_test2], y_hot_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0
    )

    # Evaluate the model
    predictions = cd_model.predict([X_test1, X_test2])
    y_pred = np.argmax(predictions, axis=3)
    y_true = y_test

    all_labels = y_true.flatten()
    all_predictions = y_pred.flatten()
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

    cd_model_name = f"CD_Simple_CBMI_{generate_short_id()}.h5"
    log_params_sim1("Task 1", " ", ' ', " ", " ", "Softmax", " ", 'Adam', EPOCHS, 'weighted_categorical_crossentropy', " ", recall, specificity, precision, f1, accuracy, "CBMI Set", 96, " ", "none", cd_model_name)

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
output_path = f'/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/FCSiam/Box_plots/boxplot_metrics_64_runs_{generate_short_id()}.png'
plt.savefig(output_path)
print(f"Box plot saved to {output_path}")
