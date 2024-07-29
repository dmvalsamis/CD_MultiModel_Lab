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


def log_params(dataset_name, model_id, modelname, lr, opt, loss, epochs, batch, dsize, trsize, tesize, vsize, tracc, trloss, valacc, valloss, testacc, testloss, normdata):


    data_dict = {'Dataset Name':dataset_name,
                'Model_ID' : model_id, 
                'Model':modelname, 
                'Learning Rate':lr, 
                'Optimizer':opt, 
                'Loss':loss, 
                'Epochs':epochs, 
                'Batch Size':batch, 
                'All':dsize, 
                'Train':trsize,
                'Test':tesize,
                'Val':vsize, 
                'Train Accuracy':tracc, 
                'Train Loss':trloss, 
                'Val Accuracy':valacc, 
                'Val Loss':valloss, 
                'Test Accuracy':testacc, 
                'Test Loss':testloss, 
                'Normalized':normdata}

    # Make data frame of above data
    df = pd.DataFrame(data_dict, index=[0])
 
    # append data frame to CSV file
    df.to_csv('pretext_task1_models.csv', mode='a', index=False, header=False)
    # print message
    print("Data logged successfully.")

