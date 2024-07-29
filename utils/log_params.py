#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:27:48 2022

@author: aleoikon
"""


# Python program to demonstrate
# writing to CSV
#import csv
import pandas as pd

def log_params(dataset_name, model_id, modelname, lr, opt, loss, epochs, batch, dsize, trsize, tesize, vsize, tracc, trloss, valacc, valloss, testacc, testloss, normdata, time):

    data_dict = {
        'Dataset Name': dataset_name,
        'Model ID' : model_id,
        'Model': modelname, 
        'Learning Rate': lr, 
        'Optimizer': opt, 
        'Loss': loss, 
        'Epochs': epochs, 
        'Batch Size': batch, 
        'All': dsize, 
        'Train': trsize,
        'Test': tesize,
        'Val': vsize, 
        'Train Accuracy': tracc, 
        'Train Loss': trloss, 
        'Val Accuracy': valacc, 
        'Val Loss': valloss, 
        'Test Accuracy': testacc, 
        'Test Loss': testloss, 
        'Normalized': normdata,
        'Duration': time                
    }

    # Define the CSV file path
    csv_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/logs/pretext_task_1_models.csv'

     # Make data frame of above data
    df = pd.DataFrame(data_dict, index=[0])
    # Append data to the CSV file
    with open(csv_path, 'a') as f:
        # If the file is empty, write the headers
        if f.tell() == 0:
            pd.DataFrame([data_dict.keys()]).to_csv(f, header=False, index=False)
        # Append data
        pd.DataFrame([data_dict.values()]).to_csv(f, header=False, index=False)


    # Print message
    print("Data logged successfully.")


def log_params_pre2(dataset_name, model_id, modelname, LEARNING_RATE, EPOCHS, BATCH_SIZE, opt, loss, dropout, decay, n_ch, IMG_HEIGHT, IMG_WIDTH, channel, method, trsize, tesize, vsize,psim,nsim):

    data_dict = {
        'Dataset Name': dataset_name,
        'Model ID' : model_id,
        'Model': modelname, 
        'Learning Rate': LEARNING_RATE, 
        'Optimizer': opt, 
        'Loss': loss, 
        'Epochs': EPOCHS, 
        'Batch Size': BATCH_SIZE, 
        'Dropout' : dropout,
        'Decay' : decay,
        'Number of Channels' : n_ch,
        'Image Height' : IMG_HEIGHT,
        'Image Width' : IMG_WIDTH,
        'Channel' : channel,
        'Method' : method,
        'Train': trsize,
        'Test': tesize,
        'Val': vsize,
        'Positive Similarity' : psim,
        'Negative Similarity' : nsim                
    }

    # Define the CSV file path
    csv_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/logs/pretext_task_2_models.csv'

     # Make data frame of above data
    df = pd.DataFrame(data_dict, index=[0])
    # Append data to the CSV file
    with open(csv_path, 'a') as f:
        # If the file is empty, write the headers
        if f.tell() == 0:
            pd.DataFrame([data_dict.keys()]).to_csv(f, header=False, index=False)
        # Append data
        pd.DataFrame([data_dict.values()]).to_csv(f, header=False, index=False)


    # Print message
    print("Data logged successfully.")

def log_params_sim1(task, downstream, module, weight, depth, acitvation, lr, opt, epochs, loss, batch, recall, specificity, precision, f_measure, accuracy, dataset, size, NORM, pretext_model,cd_model_name):

    data_dict = {'Pretext':task,
             'Downstream':downstream, 
             'Module' : module,
             'Weights': weight,
             'Depth': depth,
             'Acitvation': acitvation,
             'Epochs': epochs,
             'Optimizer': opt,
             'Learning Rate': lr,
             'Loss': loss,
             'Batch Size': batch,
             'Sensitivity/Recall':recall, 
             'Specificity':specificity, 
             'Precision':precision, 
             'F1':f_measure, 
             'Accuracy':accuracy, 
             'Set':dataset, 
             'ImageSize':size,
             'Norm':NORM,
             'Pre Model':pretext_model, 
             'CD model':cd_model_name
             
    }

    # Define the CSV file path
    csv_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/logs/new_models_results_1.csv'

     # Make data frame of above data
    df = pd.DataFrame(data_dict, index=[0])
    # Append data to the CSV file
    with open(csv_path, 'a') as f:
        # If the file is empty, write the headers
        if f.tell() == 0:
            pd.DataFrame([data_dict.keys()]).to_csv(f, header=False, index=False)
        # Append data
        pd.DataFrame([data_dict.values()]).to_csv(f, header=False, index=False)


    # Print message
    print("Data logged successfully.")



def log_params_cva(dataset_name, id, modelname, mtype, depth, dropout, decay, ImageSize, n_ch, channel, normdata):

    data_dict = {
        'Dataset Name': dataset_name,
        'ID' : id,
        'Model': modelname, 
        'Number of Channels': n_ch, 
        'Channel': channel, 
        'Model Type' : mtype,
        'Depth': depth, 
        'Dropout': dropout, 
        'Decay': decay, 
        'ImageSize': ImageSize,
        'Normalized': normdata
                      
    }

    # Define the CSV file path
    csv_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/logs/cva_downstream_models.csv'

     # Make data frame of above data
    df = pd.DataFrame(data_dict, index=[0])
    # Append data to the CSV file
    with open(csv_path, 'a') as f:
        # If the file is empty, write the headers
        if f.tell() == 0:
            pd.DataFrame([data_dict.keys()]).to_csv(f, header=False, index=False)
        # Append data
        pd.DataFrame([data_dict.values()]).to_csv(f, header=False, index=False)


    # Print message
    print("Data logged successfully.")