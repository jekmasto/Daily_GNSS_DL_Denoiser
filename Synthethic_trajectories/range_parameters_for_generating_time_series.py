#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 7 11:18:25 2023

@author: giacomo
"""

import sys ,glob, os,pickle,random
import numpy as np
import datetime as dt
from gratsid_tf_gpu_functions_SHARED import *

### Input your current directory where you have the gratsid folder
root_dir = input('Enter the root directory folder: ')

### Directory name to search for
directory_name = 'sharing_gratsid_tf_in_development'

# Iterate through the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    for dirname in dirnames:
        if dirname == directory_name:
            # Found the directory
            directory_path = os.path.join(dirpath, dirname)
            print("Directory found:", directory_path)
            break
            
if not directory_path:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_name}' not found. Be sure you have installed Gratsid")

sys.path.append(directory_path)
from gratsid_tf_gpu_functions_SHARED import *
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

########### Data directory ###########
soln_folder_path='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/sols_tables_'
components=['E/','N/']
gen_jjj = np.vectorize(lambda x,y,z: dt.date.toordinal(dt.date(x,y,z)))

sa_coeff=0
tr_coeff=0
trans_coeff=0

for c in range(len(components)):

    ############ finally build ############
    soln_folder=soln_folder_path+components[c]
    sol_path = soln_folder+name+'.npz'
    options = np.load(sol_path,allow_pickle=True)['options']
    options = options.item()
    data_cols = np.load(sol_path)['data_cols']
    
    ten_step=10
    ########## Allocate  ##########
    for i in range(len(names)):
        name=names[i]
        sol_path = soln_folder+name+'.npz'
        data_path = data_folder_path+name+'.txt'
    
        ## loading in the data and the corresponding gratsid solution table
        data = np.loadtxt(data_path)
        perm = list(np.load(sol_path,allow_pickle=True)['perm'])
        sols = list(np.load(sol_path,allow_pickle=True)['sols'])
    
        ## converting time to python datetime integer and isolating the fit directional components
        t = gen_jjj(data[:,0].astype(int),data[:,1].astype(int),data[:,2].astype(int))
        y = data[:,data_cols] ## columns 3,4,5  (in python indexing) are E,N,U
        
        signal = fit_decompose(t,y,None,options['tik_mul'], \
                sols,np.asarray(perm),options['bigTs'],options['Fs']) 
        
        table = np.vstack([perm,sols[-1][-1]])
        G, m_keys = assemble_G_return_keys(t,table,options['bigTs'],options['Fs'])

        residual,residual_val,weighted_residual_val,m \
            = single_fit_predict_tf(G_in=G,y_in=y,err_in=None,tik_mul=options['tik_mul'])
        
        ## This is the maximum coefficient for seasonal a annual amplitudes
        if max(m[np.where(m_keys==2)])>sa_coeff:
            sa_coeff=max(m[np.where(m_keys==2)])
            
        if max(m[np.where(m_keys==1)])>tr_coeff:
            tr_coeff=max(m[np.where(m_keys==1)])
        
        if 3 in m_keys:
            if max(m[np.where(m_keys==3)])>trans_coeff:
                trans_coeff=max(m[np.where(m_keys==3)])
            
        perc=i*100/len(names)
        if perc>ten_step:
            print(str(round(perc))+'%')
            ten_step+=10
            print(sa_coeff)
            
print('Max amplitude for seasonals: ',sa_coeff)
print('Max amplitude for transients: ',trans_coeff)
print('Max amplitude for trend: ',tr_coeff)
