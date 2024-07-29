#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:15:48 2022

@author: aleoikon
"""

import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')
import time

from architectures.similarity_detection import pretext_task_one_nopool,pretext_task_one_aspp

# Now you can use pretext_task_one_nopool in your script

#from tests import change_detection_noup, change_detection_noup_1x1convs
#from similarity_detection import pretext_task_one_nopool
import tensorflow
from tensorflow.keras import layers, Model
from architectures.branch import branches_triplet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import KFold


import pandas as pd
import numpy as np
import os
from tensorflow import keras
from architectures.conv_classifier import conv_classifier_two, conv_classifier_two_with_nspp,conv_classifier_two_with_aspp
from utils.layer_select import feature_selector, feature_selector_simple, transfer_learning_model #, feature_selector_task2
from utils.my_metrics import recall, accuracy, specificity, precision, f_measure, get_confusion_matrix
from utils.log_params import log_params_sim1

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import uuid
import random
from utils.weighted_cross_entropy import weighted_categorical_crossentropy
os.environ["CUDA_VISIBLE_DEVICES"]="1"

channel = 'rgb'


# Set random seed for TensorFlow
tensorflow.random.set_seed(1234)

# Set random seed for NumPy
np.random.seed(1234)




def create_rgb_onera(x,channel):
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
        b  = x[:,:,0]
        rgb = np.dstack((r,g,b))
        return(rgb)
    if channel == 'rgbvnir':
        r = x[:,:,2]
        g = x[:,:,1]
        b  = x[:,:,0]
        vnir = x[:,:,3]
        rgbvnir = np.stack((r,g,b,vnir),axis=2).astype('float')
        return(rgbvnir)
    else:
        return x
        print("NOT CORRECT CHANNELS")

def generate_short_id():
    # Generate a UUID
    unique_id = uuid.uuid4()

    # Convert UUID to a hex string and take the first 4 characters
    short_id = str(unique_id.hex)[:4]

    return short_id

def get_layer_weights(model, layer_names):
    layer_weights = {}
    for layer_name in layer_names:
        layer = model.get_layer(layer_name)
        if layer:
            weights = layer.get_weights()
            if weights:
                layer_weights[layer_name] = weights
            else:
                print("No weights found for layer:", layer_name)
        else:
            print("Layer not found:", layer_name)
    return layer_weights

def compare_weights(before_training_weights, after_training_weights):
    for layer_name in before_training_weights:
        if layer_name in after_training_weights:
            before_weights = before_training_weights[layer_name]
            after_weights = after_training_weights[layer_name]
            if len(before_weights) != len(after_weights):
                print(f"Number of weight arrays different for layer {layer_name}")
                continue
            
            all_equal = all((before_weights[i] == after_weights[i]).all() for i in range(len(before_weights)))
            if all_equal:
                print(f"Weights for layer {layer_name} are the same before and after training.")
            else:
                print(f"Weights for layer {layer_name} are different before and after training.")
        else:
            print(f"Layer {layer_name} weights not found after training.")

def print_layer_weights(model, layer_names):
    for layer_name in layer_names:
        layer = model.get_layer(layer_name)
        if layer:
            weights = layer.get_weights()
            if weights:
                print("Weights for layer", layer_name, ":", weights)
            else:
                print("No weights found for layer:", layer_name)
        else:
            print("Layer not found:", layer_name)


#----------------------------------------------------------------------------------------------------------------------------------------------------------


onera_train_target =  '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_train_data/'  
onera_test_target = '/data/valsamis_data/data/CBMI/CBMI_0.3/CBMI_0.3/NPY_dataset/aug_test_data/'

#Onera 
# onera_train_target = '/home/aleoikon/Documents/data/ssl/onera_npys/patches/downstream/train/'
# onera_test_target = '/home/aleoikon/Documents/data/ssl/onera_npys/patches/downstream/test/'


#pretext_models_df = pd.read_csv('training/pretext_tasks/pretext_task_one_models.csv')
train = pd.read_csv(onera_train_target + "dataset_train.csv")
test = pd.read_csv(onera_test_target + "dataset_test.csv")

train = train.sample(frac=1, random_state=1)
test = test.sample(frac=1, random_state=1)
print("Train Data", len(train))
print("Test Data", len(test))

NORM = True
n_ch = 3
#Load everything in memory
X_train1 = np.ndarray(shape=(len(train),96,96,n_ch))
X_train2 = np.ndarray(shape=(len(train),96,96,n_ch))
y_train =  np.ndarray(shape=(len(train),96,96))

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
(unique, counts) = np.unique(train_balance , return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies[0][1]/frequencies[1][1])

## one hot the train

y_hot_train = keras.utils.to_categorical(y_train, num_classes=2)
    
X_test1 = np.ndarray(shape=(len(test),96,96,n_ch))
X_test2 = np.ndarray(shape=(len(test),96,96,n_ch))
y_test =  np.ndarray(shape=(len(test),96,96))

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
    
## one hot the test
y_hot_test = keras.utils.to_categorical(y_test, num_classes=2)
#########
ind = random.randint(0, 1000)
plt.imshow(y_test[ind])
###############

depth = 2
dropout = 0.1
decay = 0.0001
LEARNING_RATE = 0.001
EPOCHS = 55

model_id = generate_short_id()

# #------------------------------------------------------------------------------------

cd_model = conv_classifier_two_with_aspp(depth, dropout, decay, 96, 96, n_ch)
cd_model.summary()
#plot_model(cd_model, to_file='/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/graphs/'+model_id+'_model_plot.png', show_shapes=True, show_layer_names=True)


#task1
sim_model = pretext_task_one_nopool( dropout, decay, 96, 96, n_ch)


#Load either a task1 or a task2 model
pretext_model_name = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/model_pretext1_unclouded_results.h5'
pretext_model = 'model_pretext1_unclouded_results'
#plot_model(sim_model, to_file='/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/graphs/'+model_id+'_Pretext_model_plot.png', show_shapes=True, show_layer_names=True)



#task1
sim_model.load_weights(pretext_model_name)


# Feature selection(task1)
cd_model = feature_selector_simple(depth, sim_model, cd_model)



wx = 0.1
wy = 0.2
weights = np.array([wx, wy]) #!!!!!!!!!!!!!!!!!!!!!!!! -> change
#weights = np.array([0.1,0.2])


LEARNING_RATE = 0.001
EPOCHS = 55
optimizer= Adam(learning_rate=LEARNING_RATE)
#loss='categorical_crossentropy' #weighted_bincrossentropy 'categorical_crossentropy' 'binary_crossentropy'
cd_model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(weights), metrics=['accuracy'])

BATCH_SIZE=5

    


# Before training
print("Before Training:")
# before_training_weights = get_layer_weights(cd_model, ["norm_1", "conv1_1", "norm1_1", "relu1_1", "dropout1_1", "conv2_1", "norm2_1", "relu2_1", "dropout2_1","conv3_1","norm3_1","relu3_1","dropout3_1"])



# Record start time
start_time = time.time()

history = cd_model.fit(
    [X_train1 , X_train2],
    y_hot_train,
    validation_data=([X_test1, X_test2], y_hot_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)




# # Record end time
# end_time = time.time()

# # Calculate elapsed time
# elapsed_time = end_time - start_time
# elapsed_time_minutes = elapsed_time / 60





cd_model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/CD_Simple_Levir_8192.h5'
cd_model.load_weights(cd_model_path)


###predictions
predictions = cd_model.predict([X_test1, X_test2])
y_pred = np.argmax(predictions, axis=3)




y_true = y_test


get_confusion_matrix(y_true, y_pred)
acc = accuracy(y_true,y_pred)
spec = specificity(y_true,y_pred)
rec = recall(y_true, y_pred)
prec = precision(y_true, y_pred)
f_m = f_measure(y_true, y_pred)


####see some predictions
def scaleMinMax(x):
    return ((x - np.nanpercentile(x,2)) / (np.nanpercentile(x,98) - np.nanpercentile(x,2)))


def create_rgb(x, channel):
    if channel == 'rgb':
        r = x[:,:,2]
        r = scaleMinMax(r)
        g = x[:,:,1]
        g = scaleMinMax(g)
        b  = x[:,:,0]
        b = scaleMinMax(b)
        rgb = np.dstack((r,g,b))
        return(rgb)
    else:
        one = x[:,:,0]
        one = scaleMinMax(one)
        return one        
        



# '''
# Log Params
# '''
import csv



predictions = cd_model.predict([X_test1, X_test2])
y_pred = np.argmax(predictions, axis=3)
y_true = y_test

cd_model_name = "CD_Simple_"+"CBMI_"+model_id+".h5"

model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/'
cd_model.save_weights(model_path+cd_model_name)
print("Saved model to disk")

weight_par = '[' + str(wx) + ',' + str(wy) + ']'
log_params_sim1("Task 1 (T)", "Linear", 'ASPP', weight_par, depth, "Softmax", LEARNING_RATE, 'Adam', EPOCHS, 'weighted_categorical_crossentropy', BATCH_SIZE, rec, spec, prec, f_m, acc, "CBMI Set", 96, NORM, pretext_model,"CD_Simple_Sysu_d8d2")

print("Done")

