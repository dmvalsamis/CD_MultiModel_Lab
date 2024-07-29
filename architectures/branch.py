import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2


decay = 0.0001
dropout = 0.1

def abs_diff(vects):
    x,y = vects
    result = tf.math.abs(tf.math.subtract(x,y))
    return result


def branches(IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):

    input_b = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x_b = layers.BatchNormalization()(input_b)
    
    br_conv_1_b = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name='conv1')(x_b)
    x_b = layers.Dropout(0.1, seed=1, name = 'dropout1')(br_conv_1_b)
    x_b = layers.MaxPooling2D(pool_size=(2, 2), name = 'pool1')(x_b)
    
    br_conv_2_b = layers.Conv2D(32, (3, 3), activation="relu", padding="same",name='conv2')(x_b)
    x_b = layers.Dropout(0.1, seed=1, name='dropout2')(br_conv_2_b)
    x_b = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x_b)
    
    br_conv_3_b = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name = 'conv3')(x_b)
    x_b = layers.Dropout(0.1, seed=1, name='dropout3')(br_conv_3_b)
    x_b = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x_b)
    

    branch_network_b = keras.Model(input_b, x_b, name='branch')
    
    return branch_network_b



def branches_nopool(dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):

    input_b = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x_b = layers.BatchNormalization()(input_b)
    
    br_conv_1_b = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1')(x_b)
    x_b = layers.Dropout(dropout, seed=1, name = 'dropout1')(br_conv_1_b)
    #x_b = layers.MaxPooling2D(pool_size=(2, 2), name = 'pool1')(x_b)
    
    br_conv_2_b = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2')(x_b)
    x_b = layers.Dropout(dropout, seed=1, name='dropout2')(br_conv_2_b)
    #x_b = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x_b)
    
    br_conv_3_b = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name = 'conv3')(x_b)
    x_b = layers.Dropout(dropout, seed=1, name='dropout3')(br_conv_3_b)
    #x_b = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x_b)
    

    branch_network_b = keras.Model(input_b, x_b, name='branch')
    
    return branch_network_b

def branch_cva(dropout, decay, depth, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):
  
    input_1 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
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
    
    if depth == 1:
        branch_network = keras.Model(input_1, drop1_1, name='branch')
    if depth == 2:
        branch_network = keras.Model(input_1, drop2_1, name='branch')
    if depth == 3:
        branch_network = keras.Model(input_1, drop3_1, name='branch')

    return branch_network


def create_nspp_block(input_feature, IMG_HEIGHT=96, IMG_WIDTH=96, filters=32, scales=[2, 4, 8, 16]):
    nspp_features = []

    for i, scale in enumerate(scales):
        # Pooling Block
        pooled = layers.AveragePooling2D(pool_size=(scale, scale), padding='same', name=f'pooled_{scale}')(input_feature)
        pooling_block = layers.Conv2D(filters // 4, (1, 1), padding='same', activation='relu', name=f'pooling_block_{scale}')(pooled)
        
        # Strided Convolution Block
        strided_conv = layers.SeparableConv2D(filters // 4, (3, 3), strides=(scale, scale), padding='same', name=f'strided_conv_{scale}')(input_feature)
        merged_features = layers.Add(name=f'merged_features_{scale}')([pooling_block, strided_conv])

        # Global Pooling Block
        reduced_mean = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True), name=f'reduced_mean_{scale}')(merged_features)
        pointwise_conv = layers.Conv2D(filters // 4, (1, 1), padding='same', name=f'pointwise_conv_{scale}')(reduced_mean)

        # Resize the tensor to the desired shape
        resized_feature = tf.image.resize(pointwise_conv, (IMG_HEIGHT, IMG_WIDTH), name=f'resized_feature_{scale}')

        nspp_features.append(resized_feature)

    # Channel-wise concatenation of the features from the four scales
    nspp_output = layers.Concatenate(axis=-1, name='nspp_concat')(nspp_features)
    # 1x1 Convolution to match the channel dimensions with the original input feature size
    final_nspp_output = layers.Conv2D(filters, (1, 1), padding='same', name='final_nspp_output')(nspp_output)

    return final_nspp_output





def branch_cva_with_nspp(dropout, decay, depth, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3, filters=32, scales=[2, 4, 8, 16]):
    input_1 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Depth 1
    x_1 = layers.BatchNormalization(name='norm_1')(input_1)
    conv1_1 = layers.Conv2D(filters, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1_1')(x_1)
    batch_norm1_1 = layers.BatchNormalization(name='norm1_1')(conv1_1)
    activation1_1 = layers.Activation('relu', name='relu1_1')(batch_norm1_1)
    drop1_1 = layers.Dropout(dropout, seed=1, name='dropout1_1')(activation1_1)
    output_feature = drop1_1
    
    # Depth 2
    if depth >= 2:
        conv2_1 = layers.Conv2D(filters, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv2_1')(drop1_1)
        batch_norm2_1 = layers.BatchNormalization(name='norm2_1')(conv2_1)
        activation2_1 = layers.Activation('relu', name='relu2_1')(batch_norm2_1)
        drop2_1 = layers.Dropout(dropout, seed=1, name='dropout2_1')(activation2_1)
        output_feature = drop2_1
    
    # Depth 3
    if depth == 3:
        conv3_1 = layers.Conv2D(filters, (3, 3), kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv3_1')(drop2_1)
        batch_norm3_1 = layers.BatchNormalization(name='norm3_1')(conv3_1)
        activation3_1 = layers.Activation('relu', name='relu3_1')(batch_norm3_1)
        drop3_1 = layers.Dropout(dropout, seed=1, name='dropout3_1')(activation3_1)
        output_feature = drop3_1
    
    # NSPP block is added here, after the last dropout layer based on the specified depth
    nspp_output = create_nspp_block(output_feature, IMG_HEIGHT, IMG_WIDTH, filters, scales)
    
    branch_network = keras.Model(inputs=input_1, outputs=nspp_output, name='branch_nspp')
    
    return branch_network





def branches_triplet(dropout, decay, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3):

    input_b = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x_b = layers.BatchNormalization()(input_b)
    
    br_conv_1_b = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name='conv1')(x_b)
    x_b = layers.Dropout(dropout, seed=1, name = 'dropout1')(br_conv_1_b)
 
    br_conv_2_b = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same",name='conv2')(x_b)
    x_b = layers.Dropout(dropout, seed=1, name='dropout2')(br_conv_2_b)

    br_conv_3_b = layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=l2(decay), bias_regularizer=l2(decay), padding="same", name = 'conv3')(x_b)
    x_b = layers.Dropout(dropout, seed=1, name='dropout3')(br_conv_3_b)
    
    branch_network_b = keras.Model(input_b, x_b, name='branch_triplet')
    
    return branch_network_b




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


def branch_cva_aspp(dropout, decay, depth, IMG_HEIGHT=96, IMG_WIDTH=96, IMG_CHANNELS=3, aspp_filters=32, aspp_rates=[6, 12]):
    input_1 = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    # Depth 1
    x_1 = layers.BatchNormalization(name='norm_1')(input_1)
    conv1_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), padding="same", name='conv1_1')(x_1)
    batch_norm1_1 = layers.BatchNormalization(name='norm1_1')(conv1_1)
    activation1_1 = layers.Activation('relu', name='relu1_1')(batch_norm1_1)
    drop1_1 = layers.Dropout(dropout, name='dropout1_1')(activation1_1)

    if depth == 1:
        aspp_output_1 = ASPP(drop1_1, aspp_filters, aspp_rates)
        branch_network = keras.Model(inputs=input_1, outputs=aspp_output_1, name='branch_cva_depth1')
        return branch_network
    
    # Depth 2
    conv2_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), padding="same", name='conv2_1')(drop1_1)
    batch_norm2_1 = layers.BatchNormalization(name='norm2_1')(conv2_1)
    activation2_1 = layers.Activation('relu', name='relu2_1')(batch_norm2_1)
    drop2_1 = layers.Dropout(dropout, name='dropout2_1')(activation2_1)

    if depth == 2:
        aspp_output_2 = ASPP(drop2_1, aspp_filters, aspp_rates)
        branch_network = keras.Model(inputs=input_1, outputs=aspp_output_2, name='branch_cva_depth2')
        return branch_network

    # Depth 3
    conv3_1 = layers.Conv2D(32, (3, 3), kernel_regularizer=l2(decay), padding="same", name='conv3_1')(drop2_1)
    batch_norm3_1 = layers.BatchNormalization(name='norm3_1')(conv3_1)
    activation3_1 = layers.Activation('relu', name='relu3_1')(batch_norm3_1)
    drop3_1 = layers.Dropout(dropout, name='dropout3_1')(activation3_1)
    
    aspp_output_3 = ASPP(drop3_1, aspp_filters, aspp_rates)
    branch_network = keras.Model(inputs=input_1, outputs=aspp_output_3, name='branch_cva_depth3')
    
    return branch_network




def two_branch_cva_with_aspp(dropout, decay, depth, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, aspp_filters=32, aspp_rates=[6, 12]):

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
    
    encoder_output = distance  # Output from the encoder stage

    # Apply the ASPP block after the last convolutional layer
    aspp_output = ASPP(encoder_output, aspp_filters, aspp_rates)

    # Create model
    branch_model = keras.Model(inputs=[input_1, input_2], outputs=aspp_output, name='branch')
    
    return branch_model


def two_branch_cva_with_aspp_fmaps(dropout, decay, depth, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, aspp_filters=32, aspp_rates=[6, 12]):

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

    # After applying ASPP, you already have aspp_output
    aspp_output = ASPP(encoder_output, aspp_filters, aspp_rates)
    
    # After ASPP, reducing the channels
    reduced_aspp_output = layers.Conv2D(32, (1, 1), padding="same", name='reduced_aspp_output')(aspp_output)
    reduced_aspp_output = layers.BatchNormalization()(reduced_aspp_output)
    reduced_aspp_output = layers.Activation('relu')(reduced_aspp_output)

    # Final output layer
    output = layers.Conv2D(2, (1, 1), activation="softmax", padding="same", name="output")(reduced_aspp_output)

    # Modify the model creation line to output both the ASPP feature maps and the final model output
    final_model = keras.Model(inputs=[input_1, input_2], outputs=[output, output], name='branch_with_aspp_features')

    # Define layers for which you want to see the feature maps
    layers_of_interest = ['aspp_reduced_relu', 'abs_diff_2','reduced_aspp_output','output']
    feature_map_outputs = [final_model.get_layer(name).output for name in layers_of_interest]

    # Create a new model that outputs feature maps
    feature_model = keras.Model(inputs=final_model.inputs, outputs=feature_map_outputs)

    return final_model, feature_model
    
    
