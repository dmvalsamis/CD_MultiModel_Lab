#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:08:56 2022

@author: aleoikon
"""

'''
Recall = Tp / Tp + Fn x 100
Accuracy = Tp + Tn / Tp + Fn + Fp + Tn x 100
Specificity = Tn / Tn + Fp x 100
Precision = Tp / Tp + Fp x100
F_measure = 2 x Recall x Precision / Recall + Precision

TP: True Change, Predicted Change
TN: True No Change, Predicted No Change
FP: True No Change, Predicted Change
FN: True Change, Predicted No Change
'''


import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_true = np.array([0,0,0,1,1,1])
y_pred = np.array([0,1,0,1,0,0])
confusion_matrix(y_true, y_pred, labels=[0,1])

def get_confusion_matrix(y_true, y_predicted):
    y_true = y_true.flatten()
    y_pred = y_predicted.flatten()
    print(confusion_matrix(y_true, y_pred, labels=[0,1]))
    

def recall(y_true, y_predicted):
    y_true = y_true.flatten()
    y_pred = y_predicted.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() 
    #print(confusion_matrix(y_true, y_pred, labels=[0,1]))
    recall = (tp / (tp+fn)) * 100
    return recall

def accuracy(y_true, y_predicted):
    y_true = y_true.flatten()
    y_pred = y_predicted.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() 
    accuracy = ((tp+tn)/(tp+fn+fp+tn))*100
    return accuracy

def specificity(y_true, y_predicted):
    y_true = y_true.flatten()
    y_pred = y_predicted.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    specificity = (tn / (tn+fp))*100
    return specificity

def precision(y_true, y_predicted):
    y_true = y_true.flatten()
    y_pred = y_predicted.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    precision = (tp/(tp+fp))*100
    return precision

def f_measure(y_true,y_predicted):
    y_true = y_true.flatten()
    y_pred = y_predicted.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    f = (2 * recall(y_true,y_pred) * precision(y_true,y_pred)) / (recall(y_true,y_pred) + precision(y_true,y_pred))
    return f


from sklearn.metrics import roc_curve, roc_auc_score

def get_roc(y_true,y_predicted):
    y_true = y_true.flatten()
    y_pred = y_predicted.flatten()
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_true, y_pred)
    print('roc_auc_score: ', roc_auc_score(y_true, y_pred))

    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()





