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

import pandas as pd
import numpy as np
import os
from tensorflow import keras
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp, conv_classifier_two_with_aspp
from utils.layer_select import feature_selector, feature_selector_simple, transfer_learning_model
from utils.my_metrics import recall, accuracy, specificity, precision, f_measure, get_confusion_matrix
from utils.log_params import log_params_sim1

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import uuid
import random
from utils.weighted_cross_entropy import weighted_categorical_crossentropy
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

channel = 'rgb'

# Set random seed for TensorFlow
tensorflow.random.set_seed(1234)

# Set random seed for NumPy
np.random.seed(1234)


def generate_short_id():
    # Generate a UUID
    unique_id = uuid.uuid4()

    # Convert UUID to a hex string and take the first 4 characters
    short_id = str(unique_id.hex)[:4]

    return short_id

def create_rgb_onera(x, channel):
    if channel == 'red':
        r = x[:,:,2]
        r = np.expand_dims(r, axis=2)
        return r
    if channel == 'green':
        g = x[:,:,1]
        g = np.expand_dims(g, axis=2)
        return g
    if channel == 'blue':
        b = x[:,:,0]
        b = np.expand_dims(b, axis=2)
        return b
    if channel == 'rgb':
        r = x[:,:,2]
        g = x[:,:,1]
        b = x[:,:,0]
        rgb = np.dstack((r, g, b))
        return rgb
    if channel == 'rgbvnir':
        r = x[:,:,2]
        g = x[:,:,1]
        b = x[:,:,0]
        vnir = x[:,:,3]
        rgbvnir = np.stack((r, g, b, vnir), axis=2).astype('float')
        return rgbvnir
    else:
        return x
        print("NOT CORRECT CHANNELS")

def generate_short_id():
    # Generate a UUID
    unique_id = uuid.uuid4()

    # Convert UUID to a hex string and take the first 4 characters
    short_id = str(unique_id.hex)[:4]

    return short_id

# Data Loading and Preparation

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

##### see the ration of 1 to 0s
train_balance = y_train.flatten()
(unique, counts) = np.unique(train_balance, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies[0][1] / frequencies[1][1])

# One-hot encode the combined labels
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

# Combine X and y for k-fold
X1 = np.concatenate((X_train1, X_test1), axis=0)
X2 = np.concatenate((X_train2, X_test2), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# One-hot encode the combined labels
y_hot = keras.utils.to_categorical(y, num_classes=2)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1)

depth = 2
dropout = 0.1
decay = 0.0001
LEARNING_RATE = 0.001
EPOCHS = 55
BATCH_SIZE=5


fold_no = 1
results = []

for train_index, val_index in kf.split(X1):
    print(f'Training fold {fold_no}...')
    
    # Split the data
    X_train1_fold, X_val1_fold = X1[train_index], X1[val_index]
    X_train2_fold, X_val2_fold = X2[train_index], X2[val_index]
    y_train_fold, y_val_fold = y_hot[train_index], y_hot[val_index]

    # Create a new instance of the model
    cd_model = conv_classifier_two_with_aspp(depth, dropout, decay, 96, 96, n_ch)
    
    # Load pretext model weights
    sim_model = pretext_task_one_nopool(dropout, decay, 96, 96, n_ch)
    pretext_model_name = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/model_pretext1_unclouded_results.h5'
    pretext_model = 'model_pretext1_unclouded_results'
    sim_model.load_weights(pretext_model_name)
    
    # Feature selection(task1)
    cd_model = feature_selector_simple(depth, sim_model, cd_model)
    
    wx = 0.1
    wy = 0.2
    weights = np.array([wx, wy])
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    cd_model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])
    
    # Train the model
    history = cd_model.fit(
        [X_train1_fold, X_train2_fold],
        y_train_fold,
        validation_data=([X_val1_fold, X_val2_fold], y_val_fold),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Evaluate the model
    predictions = cd_model.predict([X_val1_fold, X_val2_fold])
    y_pred = np.argmax(predictions, axis=3)
    y_true = np.argmax(y_val_fold, axis=3)
    
    acc = accuracy(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)
    f_m = f_measure(y_true, y_pred)
    
    results.append({
        'fold': fold_no,
        'accuracy': acc,
        'specificity': spec,
        'recall': rec,
        'precision': prec,
        'f1_score': f_m
    })

    weight_par = '[' + str(wx) + ',' + str(wy) + ']'
    log_params_sim1("Task 1 (T)", "Linear", 'ASPP', weight_par, depth, "Softmax", LEARNING_RATE, 'Adam', EPOCHS, 'weighted_categorical_crossentropy', BATCH_SIZE, rec, spec, prec, f_m, acc, "CBMI Set", 96, NORM, pretext_model, "CD_Simple_Sysu_d8d2")
    
    fold_no += 1

# Print results
for result in results:
    print(f"Fold {result['fold']}:")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Specificity: {result['specificity']:.4f}")
    print(f"  Recall: {result['recall']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  F1 Score: {result['f1_score']:.4f}")

# Save the model after training on all folds
model_id = generate_short_id()
cd_model_name = "CD_Simple_" + "CBMI_" + model_id + ".h5"
model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/'



cd_model.save_weights(model_path + cd_model_name)
weight_par = '[' + str(wx) + ',' + str(wy) + ']'
log_params_sim1("Task 1 (T)", "Linear", 'ASPP', weight_par, depth, "Softmax", LEARNING_RATE, 'Adam', EPOCHS, 'weighted_categorical_crossentropy', BATCH_SIZE, rec, spec, prec, f_m, acc, "CBMI Set", 96, NORM, pretext_model, "CD_Simple_Sysu_d8d2")
    
print("Saved model to disk")


print("Done")
