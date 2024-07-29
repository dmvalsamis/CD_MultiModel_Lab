from architectures.branch import branches, branches_nopool, branch_cva
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2

def abs_diff(vects):
    x,y = vects
    result = tf.math.abs(tf.math.subtract(x,y))
    return result

def pretext_task_one(IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):
    
    input_1 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    input_2 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    
    embedding_network  = branches(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    x1 = embedding_network(input_1)
    x2 = embedding_network(input_2)
    
    #absolute difference layer
    merge_layer = layers.Lambda(abs_diff)([x1,x2])
    
    #joint, a 3x3 conv layer with a relu activation function
    joint = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name='joint')(merge_layer)
    dr_joint= layers.Dropout(0.1, seed=1, name = 'dropout_joint')(joint)
    pool_joint = layers.MaxPooling2D(pool_size=(2, 2), name = 'pool_joint')(dr_joint)
    
    flatten = layers.Flatten()(pool_joint)
    dense1 = layers.Dense(128, activation="relu", name='dense1')(flatten)
    dense2 = layers.Dense(64, activation="relu", name='dense2')(dense1)
    output_layer = layers.Dense(1, activation="sigmoid", name='output')(dense2)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer, name='pretext_task1')

    return siamese

def pretext_task_one_nopool(dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):

    input_1 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    input_2 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    
    embedding_network  = branches_nopool(dropout, decay, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    x1 = embedding_network(input_1)
    x2 = embedding_network(input_2)
    
    #absolute difference layer
    merge_layer = layers.Lambda(abs_diff)([x1,x2])
    
    #joint, a 3x3 conv layer with a relu activation function
    joint = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='joint')(merge_layer)
    dr_joint= layers.Dropout(dropout, seed=1, name = 'dropout_joint')(joint)
    #pool_joint = layers.MaxPooling2D(pool_size=(2, 2), name = 'pool_joint')(dr_joint)
    
    flatten = layers.Flatten()(dr_joint)
    dense1 = layers.Dense(128, activation="relu", name='dense1')(flatten)
    dense2 = layers.Dense(64, activation="relu", name='dense2')(dense1)
    output_layer = layers.Dense(1, activation="sigmoid", name='output')(dense2)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer, name='pretext_task1')

    return siamese

def pretext_one(dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):
    input_1 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    input_2 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    
    embedding_network  = branch_cva(dropout, decay,3,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    x1 = embedding_network(input_1)
    x2 = embedding_network(input_2)
    
    #absolute difference layer
    merge_layer = layers.Lambda(abs_diff)([x1,x2])
    
    #joint, a 3x3 conv layer with a relu activation function
    joint = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='joint')(merge_layer)
    batch_norm_joint = layers.BatchNormalization(name='norm_joint')(joint)
    activation_joint = layers.Activation('relu', name='relu_joint')(batch_norm_joint)
    drop_joint = layers.Dropout(dropout, seed=1, name='dropout_joint')(activation_joint)
    
    flatten = layers.Flatten()(drop_joint)
    dense1 = layers.Dense(128, activation="relu", name='dense1')(flatten)
    dense2 = layers.Dense(64, activation="relu", name='dense2')(dense1)
    output_layer = layers.Dense(1, activation="sigmoid", name='output')(dense2)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer, name='pretext_task1')

    return siamese




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

def pretext_task_one_aspp(dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3, aspp_filters=32, aspp_rates=[6, 12]):

    input_1 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    input_2 = layers.Input((int(IMG_HEIGHT), int(IMG_WIDTH), int(IMG_CHANNELS)))
    
    embedding_network = branches_nopool(dropout, decay, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    x1 = embedding_network(input_1)
    x2 = embedding_network(input_2)
    
    # Absolute difference layer
    merge_layer = layers.Lambda(abs_diff)([x1, x2])
    
    # Insert ASPP block right after the encoder phase
    aspp_output = ASPP(merge_layer, aspp_filters, aspp_rates)
    
    # Continue with joint, a 3x3 conv layer with a relu activation function
    joint = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='joint')(aspp_output)
    dr_joint = layers.Dropout(dropout, seed=1, name='dropout_joint')(joint)
    
    flatten = layers.Flatten()(dr_joint)
    dense1 = layers.Dense(128, activation="relu", name='dense1')(flatten)
    dense2 = layers.Dense(64, activation="relu", name='dense2')(dense1)
    output_layer = layers.Dense(1, activation="sigmoid", name='output')(dense2)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer, name='pretext_task1_with_aspp')

    return siamese

