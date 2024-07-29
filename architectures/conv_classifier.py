#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:49:31 2022

@author: aleoikon
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2

def abs_diff(vects):
    x,y = vects
    result = tf.math.abs(tf.math.subtract(x,y))
    return result

def conv_classifier(depth, dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):
    input_1 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    input_2 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    ##depth 1
    x_1 = layers.BatchNormalization(name='norm_1')(input_1)
    
    conv1_1 = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1_1')(x_1)
    drop1_1 = layers.Dropout(dropout, seed=1, name = 'dropout1_1')(conv1_1)
    ##depth 2
    conv2_1 = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2_1')(drop1_1)
    drop2_1 = layers.Dropout(dropout, seed=1, name='dropout2_1')(conv2_1)
    
    ##depth 3
    conv3_1 = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv3_1')(drop2_1)
    drop3_1 = layers.Dropout(dropout, seed=1, name='dropout3_1')(conv3_1)
       
    x_2 = layers.BatchNormalization(name='norm_2')(input_2)
    ##depth 1
    conv1_2 = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1_2')(x_2)
    drop1_2 = layers.Dropout(dropout, seed=1, name = 'dropout1_2')(conv1_2)
    
    ##depth 2
    conv2_2 = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2_2')(drop1_2)
    drop2_2 = layers.Dropout(dropout, seed=1, name='dropout2_2')(conv2_2)
     
    ##depth 3
    conv3_2 = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv3_2')(drop2_2)
    drop3_2 = layers.Dropout(dropout, seed=1, name='dropout3_2')(conv3_2)
    #######################################UPSAMPLING#####################################################
    
    if depth == 1:
        distance = layers.Lambda(abs_diff)([drop1_1, drop1_2])
    if depth == 2:
        distance = layers.Lambda(abs_diff)([drop2_1, drop2_2])
    if depth == 3:
        distance = layers.Lambda(abs_diff)([drop3_1, drop3_2])
    
    
    
    conv2dsmall_1 = layers.Conv2D(32, (1,1), activation="relu", padding="same", name='conv2dsmall_1')(distance)
    conv2dsmall_2 = layers.Conv2D(16, (1,1), activation="relu", padding="same", name='conv2dsmall_2')(conv2dsmall_1)
    conv2dsmall_3 = layers.Conv2D(8, (1,1), activation="relu", padding="same", name='conv2dsmall_3')(conv2dsmall_2)    
    conv2dsmall_4 = layers.Conv2D(4, (1,1), activation="relu", padding="same", name='conv2dsmall_4')(conv2dsmall_3)
    
    output = layers.Conv2D(2,(1,1), activation = "softmax", padding="same", name="output")(conv2dsmall_4)
    
    change_detection = keras.Model([input_1, input_2], output, name='change_detection')
    
    return change_detection

def conv_classifier_two(depth, dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):
    input_1 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    input_2 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    ##depth 1
    x_1 = layers.BatchNormalization(name='norm_1')(input_1)
    
    conv1_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1_1')(x_1)
    batch_norm1_1 = layers.BatchNormalization(name='norm1_1')(conv1_1)
    activation1_1 = layers.Activation('relu', name='relu1_1')(batch_norm1_1)
    drop1_1 = layers.Dropout(dropout, seed=1, name='dropout1_1')(activation1_1)
    
    ##depth 2
    conv2_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2_1')(drop1_1)
    batch_norm2_1 = layers.BatchNormalization(name='norm2_1')(conv2_1)
    activation2_1 = layers.Activation('relu', name='relu2_1')(batch_norm2_1)
    drop2_1 = layers.Dropout(dropout, seed=1, name='dropout2_1')(activation2_1)
    
    ##depth 3
    conv3_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv3_1')(drop2_1)
    batch_norm3_1 = layers.BatchNormalization(name='norm3_1')(conv3_1)
    activation3_1 = layers.Activation('relu', name='relu3_1')(batch_norm3_1)
    drop3_1 = layers.Dropout(dropout, seed=1, name='dropout3_1')(activation3_1)
       
    x_2 = layers.BatchNormalization(name='norm_2')(input_2)
    
    ##depth 1
    conv1_2 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1_2')(x_2)
    batch_norm1_2 = layers.BatchNormalization(name='norm1_2')(conv1_2)
    activation1_2 = layers.Activation('relu', name='relu1_2')(batch_norm1_2)
    drop1_2 = layers.Dropout(dropout, seed=1, name='dropout1_2')(activation1_2)
    
    ##depth 2
    conv2_2 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2_2')(drop1_2)
    batch_norm2_2 = layers.BatchNormalization(name='norm2_2')(conv2_2)
    activation2_2 = layers.Activation('relu', name='relu2_2')(batch_norm2_2)
    drop2_2 = layers.Dropout(dropout, seed=1, name='dropout2_2')(activation2_2)
     
    ##depth 3
    conv3_2 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv3_2')(drop2_2)
    batch_norm3_2 = layers.BatchNormalization(name='norm3_2')(conv3_2)
    activation3_2 = layers.Activation('relu', name='relu3_2')(batch_norm3_2)
    drop3_2 = layers.Dropout(dropout, seed=1, name='dropout3_2')(activation3_2)
    #######################################UPSAMPLING#####################################################
    
    if depth == 1:
        distance = layers.Lambda(abs_diff)([drop1_1, drop1_2])
    if depth == 2:
        distance = layers.Lambda(abs_diff)([drop2_1, drop2_2])
    if depth == 3:
        distance = layers.Lambda(abs_diff)([drop3_1, drop3_2])
    
    
    '''
    conv2dsmall_1 = layers.Conv2D(32, (1,1), activation="relu", padding="same", name='conv2dsmall_1')(distance)
    conv2dsmall_2 = layers.Conv2D(16, (1,1), activation="relu", padding="same", name='conv2dsmall_2')(conv2dsmall_1)
    conv2dsmall_3 = layers.Conv2D(8, (1,1), activation="relu", padding="same", name='conv2dsmall_3')(conv2dsmall_2)    
    conv2dsmall_4 = layers.Conv2D(4, (1,1), activation="relu", padding="same", name='conv2dsmall_4')(conv2dsmall_3)
    '''
    conv2dsmall_1 = layers.Conv2D(32, (1,1), padding="same", name='conv2dsmall_1')(distance)
    batch_norm_1 = layers.BatchNormalization(name='norm_out_1')(conv2dsmall_1)
    activation_1 = layers.Activation('relu', name='relu_1')(batch_norm_1)
    drop_1 = layers.Dropout(dropout, seed=1, name='dropout_1')(activation_1)
    conv2dsmall_2 = layers.Conv2D(16, (1,1), padding="same", name='conv2dsmall_2')(drop_1)
    batch_norm_2 = layers.BatchNormalization(name='norm_out_2')(conv2dsmall_2)
    activation_2 = layers.Activation('relu', name='relu_2')(batch_norm_2)
    drop_2 = layers.Dropout(dropout, seed=1, name='dropout_2')(activation_2)
    conv2dsmall_3 = layers.Conv2D(8, (1,1), padding="same", name='conv2dsmall_3')(drop_2)   
    batch_norm_3 = layers.BatchNormalization(name='norm_out_3')(conv2dsmall_3)
    activation_3 = layers.Activation('relu', name='relu_3')(batch_norm_3)
    drop_3 = layers.Dropout(dropout, seed=1, name='dropout_3')(activation_3)
    conv2dsmall_4 = layers.Conv2D(4, (1,1), padding="same", name='conv2dsmall_4')(drop_3)
    batch_norm_4 = layers.BatchNormalization(name='norm_out_4')(conv2dsmall_4 )
    activation_4 = layers.Activation('relu', name='relu_4')(batch_norm_4)
    drop_4 = layers.Dropout(dropout, seed=1, name='dropout_4')(activation_4)
    
    output = layers.Conv2D(2,(1,1), activation = "softmax", padding="same", name="output")(drop_4)
    
    change_detection = keras.Model([input_1, input_2], output, name='change_detection')
    
    return change_detection








def ASPP(inputs, filters, dilation_rates):
    # 1x1 convolution
    conv_1x1 = layers.Conv2D(filters, (1, 1), padding='same', name='aspp_conv_1x1')(inputs)
    conv_1x1_bn = layers.BatchNormalization(name='aspp_conv_1x1_bn')(conv_1x1)
    conv_1x1_relu = layers.Activation('relu', name='aspp_conv_1x1_relu')(conv_1x1_bn)

    # Atrous convolutions with different dilation rates
    atrous_layers = [conv_1x1_relu]
    for idx, rate in enumerate(dilation_rates):
        atrous_conv = layers.Conv2D(filters, (3, 3), dilation_rate=rate, padding='same', name=f'aspp_conv_{rate}')(inputs)
        atrous_conv_bn = layers.BatchNormalization(name=f'aspp_conv_{rate}_bn')(atrous_conv)
        atrous_conv_relu = layers.Activation('relu', name=f'aspp_conv_{rate}_relu')(atrous_conv_bn)
        atrous_layers.append(atrous_conv_relu)
    
    # Concatenate the atrous convolutions
    concatenated = layers.Concatenate(axis=-1, name='aspp_concat')(atrous_layers)

    # Reduce the number of channels
    reduced = layers.Conv2D(filters, (1, 1), padding='same', name='aspp_reduced')(concatenated)
    reduced_bn = layers.BatchNormalization(name='aspp_reduced_bn')(reduced)
    reduced_relu = layers.Activation('relu', name='aspp_reduced_relu')(reduced_bn)
    
    return reduced_relu




def conv_classifier_two_with_aspp(depth, dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3, aspp_filters=32, aspp_rates=[6, 12]):
    input_1 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    input_2 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    ##depth 1
    x_1 = layers.BatchNormalization(name='norm_1')(input_1)
    
    conv1_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1_1')(x_1)
    batch_norm1_1 = layers.BatchNormalization(name='norm1_1')(conv1_1)
    activation1_1 = layers.Activation('relu', name='relu1_1')(batch_norm1_1)
    drop1_1 = layers.Dropout(dropout, seed=1, name='dropout1_1')(activation1_1)
    
    ##depth 2
    conv2_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2_1')(drop1_1)
    batch_norm2_1 = layers.BatchNormalization(name='norm2_1')(conv2_1)
    activation2_1 = layers.Activation('relu', name='relu2_1')(batch_norm2_1)
    drop2_1 = layers.Dropout(dropout, seed=1, name='dropout2_1')(activation2_1)
    
    ##depth 3
    conv3_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv3_1')(drop2_1)
    batch_norm3_1 = layers.BatchNormalization(name='norm3_1')(conv3_1)
    activation3_1 = layers.Activation('relu', name='relu3_1')(batch_norm3_1)
    drop3_1 = layers.Dropout(dropout, seed=1, name='dropout3_1')(activation3_1)
       
    x_2 = layers.BatchNormalization(name='norm_2')(input_2)
    
    ##depth 1
    conv1_2 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1_2')(x_2)
    batch_norm1_2 = layers.BatchNormalization(name='norm1_2')(conv1_2)
    activation1_2 = layers.Activation('relu', name='relu1_2')(batch_norm1_2)
    drop1_2 = layers.Dropout(dropout, seed=1, name='dropout1_2')(activation1_2)
    
    ##depth 2
    conv2_2 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2_2')(drop1_2)
    batch_norm2_2 = layers.BatchNormalization(name='norm2_2')(conv2_2)
    activation2_2 = layers.Activation('relu', name='relu2_2')(batch_norm2_2)
    drop2_2 = layers.Dropout(dropout, seed=1, name='dropout2_2')(activation2_2)
     
    ##depth 3
    conv3_2 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv3_2')(drop2_2)
    batch_norm3_2 = layers.BatchNormalization(name='norm3_2')(conv3_2)
    activation3_2 = layers.Activation('relu', name='relu3_2')(batch_norm3_2)
    drop3_2 = layers.Dropout(dropout, seed=1, name='dropout3_2')(activation3_2)
    #######################################UPSAMPLING#####################################################
    
    if depth == 1:
        distance = layers.Lambda(abs_diff)([drop1_1, drop1_2])
    if depth == 2:
        distance = layers.Lambda(abs_diff, name='abs_diff_2')([drop2_1, drop2_2])
    if depth == 3:
        distance = layers.Lambda(abs_diff)([drop3_1, drop3_2])
    
    encoder_output = distance  # Output from the encoder stage

    # Apply the ASPP block after the last convolutional layer
    aspp_output = ASPP(encoder_output, aspp_filters, aspp_rates)

    # After ASPP, reducing the channels
    reduced_aspp_output = layers.Conv2D(32, (1, 1), padding="same", name='reduced_aspp_output')(aspp_output)
    reduced_aspp_output = layers.BatchNormalization()(reduced_aspp_output)
    reduced_aspp_output = layers.Activation('relu')(reduced_aspp_output)

    # Final output layer
    output = layers.Conv2D(2, (1, 1), activation="softmax", padding="same", name="output")(reduced_aspp_output)
    
    # Create model
    change_detection = keras.Model(inputs=[input_1, input_2], outputs=output, name='change_detection')
    
    return change_detection