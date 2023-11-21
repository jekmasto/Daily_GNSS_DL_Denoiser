#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 14:54:08 2023

@author: giacomo
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import datetime as dt
import sys
sys.path.append('/home/giacomo/Documents/Daily_GNSS_DL_Denoiser-main/sharing_gratsid_tf_in_development/')
sys.path.append('/home/giacomo/Documents/Daily_GNSS_DL_Denoiser-main/')
sys.path.append('/home/giacomo/Documents/Daily_GNSS_DL_Denoiser-main/Prepare_data/')
from gratsid_tf_gpu_functions_SHARED import *
from funcs_4_DL_resids import *

'''
#import list of all stations
df = pd.read_csv('/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/Stations_coordinates.txt', delimiter=',',names=['station','latitude','longitude','altitude'],header=None)

components=['E','N','U']
thr=0.0015
'''
def derivative(t,data):
    """
    Compute the derivate taking into acount a variable time vector
    
    Parameters
    ----------
       t: time vector
       data: data vector

    Returns
    ----------
       derivative: with len(data)-1
    """
    if len(t)!=len(data):
        raise ValueError('The two variables have a different length')
        
    derivate=np.zeros(len(t)-1)
    for i in range(1,len(data)):
        derivate[i-1]=(data[i]- data[i-1])/(t[i]-t[i-1])
    return derivate


def find_step(signal,t,thr):
    import sys
    sys.path.append('/home/giacomo/Documents/Denoiser_GPS/Denoiser_code')
    from funcs_4_DL_resids import derivative
   
    ##### median of all step solutions #####
    step_s=np.nanmedian(np.array(signal[0]),axis=0)
    ##### velocity #####
    velocity_step=np.abs(np.concatenate([np.zeros([1]),derivative(t,step_s[:])])) 

    ##### find where velocity >thr #####
    indices=np.argwhere(velocity_step>thr)
        
    return indices,velocity_step


def find_step_single(df,thr,c=0,plot=True):

    station = df.station[random.sample(range(len(df)), 1)[0]]
    #station='UNRO'
    sol_path='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/sols_tables_'+components[c]+'/'+station+'.npz'
    options = np.load(sol_path,allow_pickle=True)['options']
    options = options.item()
    perm = list(np.load(sol_path,allow_pickle=True)['perm'])
    sols = list(np.load(sol_path,allow_pickle=True)['sols'])
    data = np.loadtxt('/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/data_bank/'+station+'.txt')
    gen_jjj = np.vectorize(lambda x,y,z: dt.date.toordinal(dt.date(x,y,z)))
    t = gen_jjj(data[:,0].astype(int),data[:,1].astype(int),data[:,2].astype(int))
    y = data[:,3+c] ## columns 3,4,5  (in python indexing) are E,N,U
    ## getting the fits
    signal = fit_decompose(t,y,None,options['tik_mul'], \
                           sols,np.asarray(perm),options['bigTs'],options['Fs']) 
    ## signal
    step_s=np.nanmedian(np.array(signal[0]),axis=0)
    velocity_step=np.abs(np.concatenate([np.zeros([1]),derivative(t,step_s[:,0])])) 
    
    ##### threshold
    indices=np.argwhere(velocity_step>thr)
    print(np.nanmedian(np.array(signal[0]),axis=0))
    if plot==True:
        fig,axes=plt.subplots(1,1,figsize=(10,5))
        ax=axes.twinx()
        ax2=axes.twinx()
        axes.plot(t,y,label='raw')
        axes.plot(t,y-np.nanmedian(np.array(signal[5]),axis=0)[:,0],label='trajectory')
        ax2.plot(t,np.nanmedian(np.array(signal[0]),axis=0)[:,0],label='step')
        ax.plot(t,velocity_step,'r',label='abs(velocity)',linewidth=0.2)
        ax.plot([t[0],t[-1]],[thr,thr],'y',label='velocity threshold')
        
        if len(indices)>0:
            ax.axvline(x=t[indices[-1]],label='step',color='g',linewidth=0.1)
        ax.legend(loc='upper right')
        [ax.axvline(x=t[i],color='g',linewidth=0.1) for i in indices[:-1]]
        ax.set_xlabel('Time [day]')
        ax.set_xlim([t[0],t[-1]])
        ax.spines['right'].set_color('red')
        ax.tick_params(axis='y', colors='red')
        ax.set_ylabel('Velocity [mm/day]',color='r')
        axes.set_ylabel('Displacement [mm]')
        axes.set_title(station+' - '+components[c])
        plt.show()
   
    return t[indices]

#t_step=find_step_single(df,thr,c=0,plot=True)




