#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:11:06 2022

@author: aleoikon
"""

def feature_selector(depth, pretext_model, cd_model, trainable=False):
    depth = depth
    print(pretext_model.summary())
    print(cd_model.summary())
    sim_model = pretext_model
    #set the weights from the conv layer in cd_model to the weights from the conv layers in the sim_model
    for i in range(len(cd_model.layers)):
        '''
        if cd_model.layers[i].name =='norm_1':
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[1].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[1].name)
        if cd_model.layers[i].name =='norm_2':
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[1].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[1].name)
        '''
        if cd_model.layers[i].name =='conv1_1' and depth >= 1:
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[2].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[2].name)
        if cd_model.layers[i].name == 'conv1_2' and depth >= 1:
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[2].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[2].name)   
        if cd_model.layers[i].name =='conv2_1' and depth >= 2:
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[4].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[4].name)
        if cd_model.layers[i].name == 'conv2_2' and depth >= 2:
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[4].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[4].name)
        if cd_model.layers[i].name =='conv3_1' and depth >= 3:
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[6].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[6].name)
        if cd_model.layers[i].name == 'conv3_2' and depth >= 3:
            cd_model.layers[i].set_weights(sim_model.layers[2].layers[6].get_weights())
            cd_model.layers[i].trainable = trainable
            print(i, cd_model.layers[i].name, sim_model.layers[2].layers[6].name)
            
            
    print(cd_model.summary())
            
    return cd_model

def feature_selector_cva(depth, source_model, branch_model):
    for i in range(len(source_model.layers)):
        #print("cva_model:", cva_model.layers[i].name)
        for j in range(len(branch_model.layers)):
            if branch_model.layers[j].name == 'norm_1' and source_model.layers[i].name == 'norm_1' and depth >= 1:
                print("Setting weights for norm_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'conv1_1' and source_model.layers[i].name == 'conv1_1' and depth >= 1:
                print("Setting weights for conv1_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'norm1_1' and source_model.layers[i].name == 'norm1_1' and depth >= 1:
                print("Setting weights for norm1_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'conv2_1' and source_model.layers[i].name == 'conv2_1' and depth >= 2:
                print("Setting weights for conv2_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'norm2_1' and source_model.layers[i].name == 'norm2_1' and depth >= 2:
                print("Setting weights for norm2_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'conv2_1' and source_model.layers[i].name == 'conv2_1' and depth >= 2:
                print("Setting weights for conv2_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'norm2_1' and source_model.layers[i].name == 'norm2_1' and depth >= 2:
                print("Setting weights for norm2_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'conv3_1' and source_model.layers[i].name == 'conv3_1' and depth >= 3:
                print("Setting weights for conv3_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            if branch_model.layers[j].name == 'norm3_1' and source_model.layers[i].name == 'norm3_1' and depth >= 3:
                print("Setting weights for norm3_1")
                branch_model.layers[j].set_weights(source_model.layers[i].get_weights())
                break
            
    return branch_model



def feature_selector_cva_with_nspp(depth, source_model, branch_model, scales=[2, 4, 8, 16]):
    for i in range(len(source_model.layers)):
        source_layer = source_model.layers[i]
        for j in range(len(branch_model.layers)):
            branch_layer = branch_model.layers[j]
            
            # Setting weights for the common layers based on depth
            if branch_layer.name == source_layer.name:
                if (branch_layer.name.startswith('conv') or branch_layer.name.startswith('norm') or branch_layer.name.startswith('dropout')) and 'nspp' not in branch_layer.name:
                    print(f"Setting weights for {branch_layer.name}")
                    branch_layer.set_weights(source_layer.get_weights())
                    break

    # Additional logic for setting weights for NSPP block layers
    for scale in scales:
        layer_names = [
            f'pooled_{scale}', 
            f'pooling_block_{scale}', 
            f'strided_conv_{scale}', 
            f'merged_features_{scale}', 
            f'reduced_mean_{scale}', 
            f'pointwise_conv_{scale}', 
            f'resized_feature_{scale}'
        ]
        
        for name in layer_names:
            try:
                branch_layer = branch_model.get_layer(name)
                source_layer = source_model.get_layer(name)
                print(f"Setting weights for {name}")
                branch_layer.set_weights(source_layer.get_weights())
            except ValueError as e:
                print(f"Layer {name} not found in one of the models: {e}")

    # Ensure the final NSPP output layer weights are also transferred
    try:
        branch_layer = branch_model.get_layer('final_nspp_output')
        source_layer = source_model.get_layer('final_nspp_output')
        print("Setting weights for final_nspp_output")
        branch_layer.set_weights(source_layer.get_weights())
    except ValueError as e:
        print(f"Layer final_nspp_output not found in one of the models: {e}")

    return branch_model


def feature_selector_cva_aspp(depth, source_model, branch_model):
    # Names of the layers to be copied, you might need to adjust based on your exact model architecture
    layer_names = [f'norm{d}_1' for d in range(1, depth + 1)] + \
                  [f'conv{d}_1' for d in range(1, depth + 1)]

    for layer_name in layer_names:
        source_layer = source_model.get_layer(layer_name)
        try:
            branch_layer = branch_model.get_layer(layer_name)
            branch_layer.set_weights(source_layer.get_weights())
            print(f"Setting weights for {layer_name}")
        except ValueError as e:
            print(f"Could not set weights for layer {layer_name}: {e}")
            continue

    # If the ASPP block is present in the branch model, copy its weights as well
    aspp_layer_names = ['aspp_concat', 'final_nspp_output']
    for layer_name in aspp_layer_names:
        if depth == 3:  # Assuming ASPP is only added at depth 3
            try:
                source_layer = source_model.get_layer(layer_name)
                branch_layer = branch_model.get_layer(layer_name)
                branch_layer.set_weights(source_layer.get_weights())
                print(f"Setting weights for {layer_name}")
            except ValueError as e:
                print(f"Could not set weights for layer {layer_name}: {e}")
                continue

    return branch_model

def two_feature_selector_cva_aspp(depth, source_model, branch_model):
    
    source_layers = {layer.name: layer for layer in source_model.layers}
    branch_layers = {layer.name: layer for layer in branch_model.layers}

    # List of layers to copy from source_model to branch_model
    layers_to_copy = [
        'norm_1', 'conv1_1', 'norm1_1', 'relu1_1', 'dropout1_1',
        'norm_2', 'conv2_1', 'norm2_1', 'relu2_1', 'dropout2_1',
        'norm_3', 'conv3_1', 'norm3_1', 'relu3_1', 'dropout3_1',
        # Add the names of layers within the ASPP block here
        'aspp_conv_1x1', 'aspp_conv_1x1_bn', 'aspp_conv_1x1_relu',
        'aspp_conv_6', 'aspp_conv_6_bn', 'aspp_conv_6_relu',
        'aspp_conv_12', 'aspp_conv_12_bn', 'aspp_conv_12_relu',
        'aspp_conv_18', 'aspp_conv_18_bn', 'aspp_conv_18_relu',
        'aspp_concat', 'aspp_reduced', 'aspp_reduced_bn', 'aspp_reduced_relu'
    ]

    for layer_name in layers_to_copy:
        if layer_name in branch_layers and layer_name in source_layers:
            branch_layer = branch_layers[layer_name]
            source_layer = source_layers[layer_name]
            branch_layer.set_weights(source_layer.get_weights())
            print(f"Transferred weights for {layer_name}")
        else:
            print(f"Layer {layer_name} not found in both models.")

    return branch_model
