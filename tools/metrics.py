#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:20:03 2018

@author: theoestienne
"""
import pandas as pd
from sklearn import metrics
import numpy as np
import SimpleITK as sitk
import os


def menisque_metrics(reference_path, prediction_path):

    pred = pd.read_csv(prediction_path, index_col=0)
    ref = pd.read_csv(reference_path, index_col=0)

    assert pred.shape[0] == ref.shape[0]
    assert pred.shape[1] == 5

    pred = pred.loc[ref.index]
    pred = pred.fillna(0.5)

    # Fissure
    ref['Fissure'] = np.logical_or(ref['Corne anterieure'], ref['Corne posterieure'])

    y_true = ref['Fissure']
    y_pred = pred['Fissure']

    auc_detection = metrics.roc_auc_score(y_true, y_pred)

    # Localisation
    y_true = ref.loc[ref['Fissure'] == True,'Corne anterieure']
    y_pred = pred.loc[ref['Fissure'] == True,'Corne anterieure']

    auc_position = metrics.roc_auc_score(y_true, y_pred)

    # Orientation
    ante = ref.loc[ref['Fissure'] == True,'Orientation anterieure'] == 'Horizontale' 
    poste = ref.loc[ref['Fissure'] == True,'Orientation posterieure'] == 'Horizontale'

    y_true = np.logical_or(ante,poste) 
    y_pred = pred.loc[ref['Fissure'] == True,'Orientation horizontale']

    auc_orientation = metrics.roc_auc_score(y_true, y_pred)

    total_score = 0.4 * auc_detection + 0.3 * auc_position + 0.3 * auc_orientation

    return total_score

# reference_path = '/home/theoestienne/Documents/JFR/menisque/menisque_train_set.csv'
# prediction_path = '/home/theoestienne/Documents/JFR/menisque/menisque_exemple.csv' 

# print(menisque_metrics(reference_path, prediction_path))

