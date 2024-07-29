#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 07:34:39 2022

@author: aleoikon
"""
import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')

import os
from tensorflow.keras import optimizers, losses
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import feature_scaling, create_rgb
from utils.log_params import log_params_pre2
from architectures.model_triplet_loss import pretext_task_2_model
import uuid

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def generate_short_id():
    # Generate a UUID
    unique_id = uuid.uuid4()

    # Convert UUID to a hex string and take the first 4 characters
    short_id = str(unique_id.hex)[:4]

    return short_id


task1_path = '/data/aleoikon_data/change_detection/ssl/s2mtcp/patches/task1/'
task2_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/training/pretext_tasks/'

dropout = 0.1
decay = 0.0001
n_ch = 4
IMG_HEIGHT = 96
IMG_WIDTH = 96
channel = 'rgbvnir'
method = 'STAND'

siamese_model, embedding = pretext_task_2_model(dropout, decay, IMG_HEIGHT, IMG_WIDTH, n_ch)

embedding.summary()

task1_df = pd.read_csv(os.path.join(task1_path, 'dataset_unclouded.csv'))
task2_df = pd.read_csv(task2_path+'task2.csv', dtype=str)

train = task2_df.sample(frac=0.85,random_state=1)
validation = task2_df.drop(train.index)
test = validation.sample(frac = 0.33, random_state=1)
validation = validation.drop(test.index)

# Length of the train dataframe
train_length = len(train)

# Length of the validation dataframe
validation_length = len(validation)

# Length of the test dataframe
test_length = len(test)

print("Data", len(task2_df))
print("85% of Data = Train", train_length)
print("10% of Data = Validation", validation_length)
print("5% of Data = Test", test_length)

X_train_an = np.ndarray(shape=(len(train),96,96,n_ch))
X_train_p = np.ndarray(shape=(len(train),96,96,n_ch))
X_train_n = np.ndarray(shape=(len(train),96,96,n_ch))
X_test_an = np.ndarray(shape=(len(test),96,96,n_ch))
X_test_p = np.ndarray(shape=(len(test),96,96,n_ch))
X_test_n = np.ndarray(shape=(len(test),96,96,n_ch))
X_val_an = np.ndarray(shape=(len(validation),96,96,n_ch))
X_val_p = np.ndarray(shape=(len(validation),96,96,n_ch))
X_val_n = np.ndarray(shape=(len(validation),96,96,n_ch))

pos = 0
for index in train.index:
    img1 = np.load(task1_path + train['Anchor'][index])
    img2 = np.load(task1_path+ train['Positive'][index])
    img3 = np.load(task1_path+ train['Negative'][index])
    X1 = create_rgb(img1,channel)
    X2 = create_rgb(img2, channel)
    X3 = create_rgb(img3, channel)
    X1 = feature_scaling(X1, method)
    X2 = feature_scaling(X2, method)
    X3 = feature_scaling(X3, method)
    X_train_an[pos] = X1
    X_train_p[pos] = X2
    X_train_n[pos] = X3
    pos += 1
    
pos = 0
for index in test.index:
    img1 = np.load(task1_path + test['Anchor'][index])
    img2 = np.load(task1_path+ test['Positive'][index])
    img3 = np.load(task1_path+ test['Negative'][index])
    X1 = create_rgb(img1,channel)
    X2 = create_rgb(img2, channel)
    X3 = create_rgb(img3, channel)
    X1 = feature_scaling(X1, method)
    X2 = feature_scaling(X2, method)
    X3 = feature_scaling(X3, method)
    X_test_an[pos] = X1
    X_test_p[pos] = X2
    X_test_n[pos] = X3
    pos += 1

pos = 0
for index in validation.index:
    img1 = np.load(task1_path + validation['Anchor'][index])
    img2 = np.load(task1_path + validation['Positive'][index])
    img3 = np.load(task1_path+ validation['Negative'][index])
    X1 = create_rgb(img1,channel)
    X2 = create_rgb(img2, channel)
    X3 = create_rgb(img3, channel)
    X1 = feature_scaling(X1, method)
    X2 = feature_scaling(X2, method)
    X3 = feature_scaling(X3, method)
    X_val_an[pos] = X1
    X_val_p[pos] = X2
    X_val_n[pos] = X3
    pos += 1
    
    
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 5
siamese_model.compile(optimizer=optimizers.Adam(LEARNING_RATE))

history = siamese_model.fit(
    [X_train_an, X_train_p, X_train_n],
    batch_size = BATCH_SIZE,
    epochs=EPOCHS, 
    validation_data=[X_val_an, X_val_p, X_val_n])

#here

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

from datetime import date

today = date.today()
print("Today's date:", today)

str(today)

from datetime import datetime, date

date_now = date.today()
current_date = date_now.strftime("%d%m%y")
now = datetime.now()
current_time = now.strftime("%H%M%S")

model_id = generate_short_id()

model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/'
model_name= "Pretext_Task_2_Model_"+"S2MTCP_"+model_id+'.h5'
embedding.save_weights(model_path+model_name)

log_params_pre2("S2MTCP", model_id, model_name, LEARNING_RATE, EPOCHS, BATCH_SIZE, 'Adam', "Triple_Loss", dropout, decay, n_ch, IMG_HEIGHT, IMG_WIDTH, channel, method, train_length, test_length, validation_length)

#evaluation.
#model_name='060722072740_pretext_task_2_(branch)rgbvnir_STAND_0.1_0.0001_0.001_20.h5'
embedding.load_weights(model_path+model_name)

positive_similarity = losses.cosine_similarity(
    embedding(X_test_an), embedding(X_test_p)
).numpy().mean()

negative_similarity = losses.cosine_similarity(
    embedding(X_test_an), embedding(X_test_n)
).numpy().mean()

# Print the outcomes
print("Positive Similarity:", positive_similarity)
print("Negative Similarity:", negative_similarity)

log_params_pre2("S2MTCP", model_id, model_name, LEARNING_RATE, EPOCHS, BATCH_SIZE, 'Adam', "Triple_Loss", dropout, decay, n_ch, IMG_HEIGHT, IMG_WIDTH, channel, method, train_length, test_length, validation_length,positive_similarity,negative_similarity)


print("done")