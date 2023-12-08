#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  15 16:29:18 2023

@author: giacomo
"""
import sys, os,random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

################## GPU OFF ##################
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")


########################################
# SET A DATASET
input_length=61
cd='/home/giacomo/Documents/DATA_synthetics_E_N_'+str(input_length)
os.chdir(cd)

datasets=['TRAINING','VALIDATION','TESTING']
XTv=np.load(cd+'/'+datasets[1]+'_tensor_step.npz')['X'] 
YTv=np.load(cd+'/'+datasets[1]+'_tensor_step.npz')['H'] 
YTvg=np.load(cd+'/'+datasets[1]+'_tensor_gratsid.npz')['Y']

if np.isnan(XTv).any():
    print("There are NaN values in the validation X tensor.")
if np.isnan(YTv).any():
    print("There are NaN values in the validation Y tensor.")

XTr=np.load(cd+'/'+datasets[0]+'_tensor_step_AUGUMENTED.npz')['X'] 
YTr=np.load(cd+'/'+datasets[0]+'_tensor_step_AUGUMENTED.npz')['H']

if np.isnan(XTr).any():
    print("There are NaN values in the training X tensor.")
if np.isnan(YTr).any():
    print("There are NaN values in the training Y tensor.")
