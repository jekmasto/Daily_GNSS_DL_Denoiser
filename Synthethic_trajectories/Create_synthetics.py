#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar Feb 21 19:53:25 2023

@author: jon - giacomo
"""

import random,sys,os,glob
sys.path.append('/home/giacomo/Documents/Synthetic_dataset/code/')
sys.path.append('/home/giacomo/Documents/Denoiser_GPS/Denoiser_code/')
from create_time_series import synth_series
from funcs_4_DL_resids import id_names_txt,elongate_and_interpolate
import numpy as np
import matplotlib.pyplot as plt

def split_nan(x):
    """
    create groups of non nans elements
    """
    return np.split(x, np.where(np.diff(np.isnan(x), prepend=True))[0])[1::2]

def randomize_noise(r):
    '''
    Transforming the original data into the frequency domain
    randomizing the phase and converting the data back into the time domain.
    '''
    ts_fourier  = np.fft.rfft(r)
    random_phases = np.exp(np.random.uniform(0,np.pi,int(len(r)/2)+1)*1.0j)
    ts_fourier_new = ts_fourier*random_phases
    return np.fft.irfft(ts_fourier_new)

def generate_syntethics_real_noise(n,soln_folder_path,save_folder_path,sec_rateA,offsetA,decay_onA,sto_mulA,components,Outliers_flag):
    """
    Generate n syntethic time series
    
    Parameters
    ----------
        n: number of time series to generate
        soln_folder_path: folder of residual
        save_folder_path: save folder
        sec_rateA: plausible trends
        offsetA: plausible offsets
        decay_onA: If decay is on (=1), the largest step will also have a logarithmic decay
        components: list of components
        Outliers_flag: boolean, if True add outliers
    
    Returns
    ----------
    """
    print('Number of time series to create: '+str(n))
    names=id_names_txt(soln_folder_path+'E')
    examples=random.choices(names, k=n)
    ### Remove prexisting files
    if os.path.isdir(save_folder_path):
        for f in os.listdir(save_folder_path):
            os.remove(os.path.join(save_folder_path, f))
    else:
        os.mkdir(save_folder_path)

    for i in range(n):
        es=examples[i]
        ### random component 
        comp = random.sample(list(components), 1)[0] 
        
        in_path = soln_folder_path+'/'+comp+'/'+es+'.txt'
        tdr = np.loadtxt(in_path)
        t_in = tdr[:,0] #time vector
        r_in = tdr[:,2] #residuals vector [NOISE]
        
        ### Create time array
        t_in=t_in-t_in[0]
        t_in=t_in+1
        t_in=t_in.astype('int')
        t=np.arange(1,t_in[-1]+1,1,dtype=int)
        
        ### stochastic trend and offset
        sec_rate = random.sample(list(sec_rateA), 1)[0] 
        offset=random.sample(list(offsetA), 1)[0]
        ### stochasticity of seasonals
        sto_mul = random.sample(list(sto_mulA), 1)[0]
        ### If decay is on (=1), the largest step will also have a logarithmic decay
        decay_on=random.sample(list(decay_onA), 1)[0]
        
        NN_full,y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \     
        step_locs_out, step_vals, frac_hh_mm_ss,boolean_outliers = synth_series(t,seas_freqs,seas_amp,sto_mul, \
            max_nsteps,magnitude_steps,GR,noise_level,offset,sec_rate,num_arctan,max_Aarctan, \
            max_Darctan,num_gau,max_Agau,max_Dgau,decay_on,Adec,Tdec,mc_on,Outliers_flag, \
            best_dist_percentage,best_dist_amplitudes,Nans_flag=False)
        
        step_times = step_locs_out
        #step_fracs = frac_hh_mm_ss[:,0]
        #known_steps = np.hstack([step_times[:,None],step_fracs[:,None]])        
         
        ### put gaps from the residuals
        ind=np.setdiff1d(np.union1d(t, t_in), np.intersect1d(t, t_in))
        NN_full[ind]=np.nan
        NN_full = y_full[~np.isnan(NN_full)]

        ### Putting nans in the generated time series 
        if np.isnan(t_in).any():
            nan_groups=split_nan(t_in) #nan groups
            for gr in nan_groups:
                indices=np.nonzero(np.in1d(t_in,gr))[0]
                NN_full[indices]=np.nan
        
        
        assert len(NN_full)==len(r_in)

        ### randomize the phase of the noise
        r_in=randomize_noise(r_in)

        ### sum generate signals and residuals [Noise]
        if len(NN_full)==len(r_in):
            d_in=NN_full+r_in
        else:
            d_in=NN_full[:-1]+r_in
            t_in=t_in[:-1]
            ###  because otherwise indexes would be wrong
            #step_times=np.array([st+1 for st in step_times])

        ### interpolating only if there are missing gaps (nans values inside)
        if np.isnan(r_in).any() or len(np.argwhere(np.diff(t_in)>1)):
            t,d,r,N = elongate_and_interpolate(t_in,d_in,r_in,r_in,max_gap)
        else:
            t,d,r=t_in,d_in,r_in
    
        ### Create 0-1 target array ###
        targety=np.zeros(len(r))
        stepind=[]
        
        ### if the step is inside a gap, put the step to the closest time point
        if len(step_times)>0:
            for ii in range(len(step_times)):
                if step_times[ii] in t:
                    stepind.append(np.argwhere(t==step_times[ii])[0][0])
                else:
                    stepind.append(np.argwhere(t==t.flat[np.abs(t - step_times[ii]).argmin()])[0][0])
        
        #stepind=[stepin+1 for stepin in stepind]
        stepind=np.array(stepind)
        #if the step is outside the time vector
        to_delete=np.argwhere(stepind>=len(targety))
        if len(to_delete)>0:
            np.delete(stepind,to_delete)

        targety[list(stepind)]=1
        
        ### put nan in the targets
        for ind in np.argwhere(np.isnan(np.array(r))):
            targety[ind[0]]=np.nan
        
        out = np.vstack([t,d,r,targety]).T
        ######  Save raw data and residuals ######
        ### Save
        out_path=save_folder_path+'/'+str(i)+'.txt'
        np.savetxt(fname=out_path,X=out)
    return 

def generate_syntethics_gan_noise(n,soln_folder_path,save_folder_path,sec_rateA,offsetA,decay_onA,sto_mulA,Outliers_flag):
    """
    Generate n syntethic time series using the noise generated by the GAN
    
    Parameters
    ----------
        n: number of time series to generate
        soln_folder_path: folder of residual
        save_folder_path: save folder
        sec_rateA: plausible trends
        offsetA: plausible offsets
        decay_onA: If decay is on (=1), the largest step will also have a logarithmic decay
        Outliers_flag: boolean, if True add outliers
        Nans_flag: boolean, if True add Nans
    
    Returns
    ----------
    """
    print('Number of time series to create: '+str(n))

    examples=np.load(soln_folder_path)
    examples=np.array(random.choices(examples, k=n))
    print(examples.shape)

    ### Remove prexisting files
    if os.path.isdir(save_folder_path):
        for f in os.listdir(save_folder_path):
            os.remove(os.path.join(save_folder_path, f))
    else:
        os.mkdir(save_folder_path)

    for i in range(n):
        r_in=examples[i,:]
         
        ### Create time array
        t=np.arange(1,r_in.shape[0]+1,1,dtype=int)
        
        ### stochastic trend and offset
        sec_rate = random.sample(list(sec_rateA), 1)[0] 
        offset=random.sample(list(offsetA), 1)[0]
        ### stochasticity of seasonals
        sto_mul = random.sample(list(sto_mulA), 1)[0]
        ### If decay is on (=1), the largest step will also have a logarithmic decay
        decay_on=random.sample(list(decay_onA), 1)[0]

        NN_full,y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \     
        step_locs_out, step_vals, frac_hh_mm_ss,boolean_outliers = synth_series(t,seas_freqs,seas_amp,sto_mul, \
            max_nsteps,magnitude_steps,GR,noise_level,offset,sec_rate,num_arctan,max_Aarctan, \
            max_Darctan,num_gau,max_Agau,max_Dgau,decay_on,Adec,Tdec,mc_on,Outliers_flag, \
            best_dist_percentage,best_dist_amplitudes,Nans_flag)
        
        step_times = step_locs_out
        #step_fracs = frac_hh_mm_ss[:,0]
        #known_steps = np.hstack([step_times[:,None],step_fracs[:,None]])        
        
        assert len(NN_full)==len(r_in)
        
        #Noise+trajectory
        d_in=NN_full+r_in
    
        ### Create 0-1 target array ### for the step model
        targety=np.zeros(len(r_in))
        stepind=[]
        
        ### if the step is inside a gap, put the step to the closest time point
        if len(step_times)>0:
            for ii in range(len(step_times)):
                if step_times[ii] in t:
                    stepind.append(np.argwhere(t==step_times[ii])[0][0])
                else:
                    stepind.append(np.argwhere(t==t.flat[np.abs(t - step_times[ii]).argmin()])[0][0])
        
        #stepind=[stepin+1 for stepin in stepind] #if the target is not the onset 
        stepind=np.array(stepind)
        #if the step is outside the time vector
        to_delete=np.argwhere(stepind>=len(targety))
        if len(to_delete)>0:
            np.delete(stepind,to_delete)

        targety[list(stepind)]=1
        
        out = np.vstack([t,d_in,r_in,targety]).T
        ######  Save raw data and residuals ######
        ### Save
        out_path=save_folder_path+'/'+str(i)+'.txt'
        np.savetxt(fname=out_path,X=out)
    return 



max_gap=5
  
################ if generate_syntethics_real_noise ################
#soln_folder_path='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/t_disps_resids/'

################ if generate_syntethics_gan_noise ################
soln_folder_path='/home/giacomo/Documents/S_NEW/U_TS_residuals.npy'
#soln_folder_path='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/Residuals_tensors/E_N_Fake_residuals_3years.npy'

save_folder_path='/home/giacomo/Documents/S_NEW/t_disps_residsF_U'

### Numbers of synthetic to generate
n=20000

################ if generate_syntethics_real_noise ################
#components=['E','N','U']
#generate_syntethics(n,soln_folder_path,save_folder_path,sec_rateA,offsetA,decay_onA,sto_mulA,components)

################ if generate_syntethics_gan_noise ################
generate_syntethics_gan_noise(n,soln_folder_path,save_folder_path,sec_rateA,offsetA,decay_onA,sto_mulA,Outliers_flag, \
            best_dist_percentage,best_dist_amplitudes,Nans_flag)


