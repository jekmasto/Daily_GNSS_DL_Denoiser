"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Function for generating the Fake GNSS daily displacement time-series

@author: jon
"""


import numpy as np
from numpy import matlib
import random

def synth_series(x,seas_freqs,seas_amp,sto_mul,max_nsteps,max_step_size,noise_level,offset,sec_rate, \
    num_arctan,max_Aarctan,max_Darctan,num_gau,max_Agau,max_Dgau,decay_on,Adec,Tdec,mc_on):

    # initializing y
    y = np.zeros(x.shape)
    
    #synthesizing the seasonal component of the signal
    coeff_rand = seas_amp*np.random.randn(seas_freqs.size,2)
    rand_sign = np.sign(np.random.rand(seas_freqs.size,2))
    y_seas = np.zeros(x.shape)
    y_seas_sto = np.zeros(x.shape)
    for i in range(coeff_rand.shape[0]):
        y_new = np.zeros(x.shape)
        y_add_sin = np.sin(x/(365.25/seas_freqs[i])*2*np.pi)
        y_add_cos = np.cos(x/(365.25/seas_freqs[i])*2*np.pi)
        a1_spread = sto_mul*coeff_rand.mean()*np.random.rand(1);
        b_spread = sto_mul*coeff_rand.mean()*np.random.rand(1);
        b_sto = b_spread*np.random.randn(x.size)
        a1_sto = a1_spread*np.random.randn(x.size)
        counter = 0
        each_side = 180
        while counter < 2:
            b_sto_new = np.zeros(b_sto.size);
            a1_sto_new = np.zeros(a1_sto.size)
            for j in range(b_sto.size):
                pp = np.arange((j-each_side),(j+each_side+1))
                keep = (pp > 0)*(pp<=b_sto.size-1)
                pp = pp[keep];
                b_sto_new[j] = b_sto[pp].mean()
                a1_sto_new[j] = a1_sto[pp].mean()
            a1_sto = a1_sto_new;
            b_sto = b_sto_new;
            counter = counter + 1
        y_new = y_new + coeff_rand[i,0]*y_add_sin
        y_new = y_new + coeff_rand[i,1]*y_add_cos
        y_new_sto = (1+a1_sto/coeff_rand[i,1])*y_new + np.multiply(b_sto,y_add_sin)
        y_seas = y_seas + y_new
        y_seas_sto = y_seas_sto + y_new_sto

    # Making the noise (as a fraction of the seasonal amplitude)
    amp_range = np.max(y_seas)-np.min(y_seas)
    y_noise = np.random.rand(1)[0]*noise_level*amp_range*np.random.randn(x.size)

    # making the steps in the data
    nsteps = np.round(np.random.rand(1)*max_nsteps)[0].astype(int)
    step_locs = np.random.permutation(x.size)
    y_steps = np.zeros(x.shape)
    day_frac = np.array([[i] for i in np.random.uniform(low=0.2, high=0.7, size=(nsteps,))]) #np.random.rand(nsteps,1) to simplify
    hh = np.floor(day_frac*24)
    t_rem = day_frac*24-hh
    mm = np.floor(t_rem*60)
    t_rem = t_rem*60-mm
    ss = t_rem*60
    frac_hh_mm_ss = np.hstack((np.hstack((np.hstack((day_frac,hh)),mm)),ss))
    step_vals = np.zeros((nsteps))
    y_steps = np.zeros(x.shape)

    if nsteps > 0:
        for i in range(nsteps):
            step_vals[i]= [random.uniform(-2*max_step_size,-0.005),random.uniform(0.005,2*max_step_size)][random.randrange(2)]
            #step_vals[i] = 2*max_step_size*(np.random.rand(1)-0.5)
            y_steps[step_locs[i]:] = y_steps[step_locs[i]:] + (1 - day_frac[i])*step_vals[i]
            y_steps[(step_locs[i]+1):] = y_steps[(step_locs[i]+1):] + (day_frac[i])*step_vals[i]
            step_locs_out = step_locs[0:(nsteps)]
    else:
        step_vals = np.zeros(0)
        step_locs_out = np.zeros(0)


    # Adding the arctan shapes
    slow_slip_all = np.zeros((num_arctan,x.size))
    for i in range(num_arctan):
        #tt = np.random.permutation(x.size)
        if i == 0:
            t_onset = int(x.size/2)
            slow_slip = np.arctan((x-t_onset)/(max_Darctan*np.random.rand(1)))
            slow_slip = slow_slip - np.min(slow_slip)
            slow_slip_all[i,:] = np.sign(np.random.rand(1)-0.5)*max_Aarctan*(slow_slip/np.max(slow_slip))
        if i == 1:
            t_onset = int(0.95*x.size)
            slow_slip = np.arctan((x-t_onset)/(max_Darctan*np.random.rand(1)))
            slow_slip = slow_slip - np.min(slow_slip)
            slow_slip_all[i,:] = np.sign(np.random.rand(1)-0.5)*max_Aarctan*(slow_slip/np.max(slow_slip))
        
    y_arctan = slow_slip_all.sum(axis=0)

    # Adding the Gaussian shapes
    gau_all = np.zeros((num_gau,x.size))
    for i in range(num_gau):
        tt = np.random.permutation(x.size)
        t_onset = tt[0]
        my_gau = max_Agau*np.random.rand(1)*np.exp(-((x-t_onset)**2) / \
        (2*(np.random.rand(1)*max_Dgau)**2))
        gau_all[i,:] = np.sign(np.random.rand(1)-0.5)*my_gau
    y_gau = gau_all.sum(axis=0)

    # Adding a decay at the location of the largest step
    if decay_on == 1 and nsteps > 0:
        pp = np.where(np.abs(step_vals) == np.max(abs(step_vals)))
        pp = pp[0][0]
        x_dec = np.zeros(x.size)
        x_dec[(step_locs_out[pp]):] = np.arange(1,(x_dec[(step_locs_out[pp]):]).size+1,1)-1
        y_dec = Adec*np.log10(1+x_dec/Tdec)
    else:
        y_dec = np.zeros(x.size)



    # Making the linear trend plus offset
    if mc_on == 1:
        y_sec = sec_rate*x + offset
    else:
        y_sec = np.zeros(x.size)

    # Combining the signals
    y_full = y_sec + y_seas_sto + y_steps + y_arctan + y_gau + y_dec + y_noise;

    return y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \
    step_locs_out, step_vals, frac_hh_mm_ss

def synth_series_2(x,seas_freqs,seas_amp,sto_mul,max_nsteps,max_step_size,noise_level,offset,sec_rate, \
    num_arctan,max_Aarctan,max_Darctan,num_gau,max_Agau,max_Dgau,decay_on,Adec,Tdec,mc_on):
    
    # initializing y
    y = np.zeros(x.shape)
    
    #synthesizing the seasonal component of the signal
    coeff_rand = seas_amp*np.random.randn(seas_freqs.size,2)
    rand_sign = np.sign(np.random.rand(seas_freqs.size,2))
    y_seas = np.zeros(x.shape)
    y_seas_sto = np.zeros(x.shape)
    for i in range(coeff_rand.shape[0]):
        y_new = np.zeros(x.shape)
        y_add_sin = np.sin(x/(365.25/seas_freqs[i])*2*np.pi)
        y_add_cos = np.cos(x/(365.25/seas_freqs[i])*2*np.pi)
        a1_spread = sto_mul*coeff_rand.mean()*np.random.rand(1);
        b_spread = sto_mul*coeff_rand.mean()*np.random.rand(1);
        b_sto = b_spread*np.random.randn(x.size)
        a1_sto = a1_spread*np.random.randn(x.size)
        counter = 0
        each_side = 180
        while counter < 2:
            b_sto_new = np.zeros(b_sto.size);
            a1_sto_new = np.zeros(a1_sto.size)
            for j in range(b_sto.size):
                pp = np.arange((j-each_side),(j+each_side+1))
                keep = (pp > 0)*(pp<=b_sto.size-1)
                pp = pp[keep];
                b_sto_new[j] = b_sto[pp].mean()
                a1_sto_new[j] = a1_sto[pp].mean()
            a1_sto = a1_sto_new;
            b_sto = b_sto_new;
            counter = counter + 1
        y_new = y_new + coeff_rand[i,0]*y_add_sin
        y_new = y_new + coeff_rand[i,1]*y_add_cos
        y_new_sto = (1+a1_sto/coeff_rand[i,1])*y_new + np.multiply(b_sto,y_add_sin)
        y_seas = y_seas + y_new
        y_seas_sto = y_seas_sto + y_new_sto

    # Making the noise (as a fraction of the seasonal amplitude)
    amp_range = np.max(y_seas)-np.min(y_seas)
    y_noise = np.random.rand(1)[0]*noise_level*amp_range*np.random.randn(x.size)

    # making the steps in the data
    nsteps = np.round(np.random.rand(1)*max_nsteps)[0].astype(int)
    step_locs = np.random.permutation(x.size)
    y_steps = np.zeros(x.shape)
    day_frac = np.random.rand(nsteps,1)
    hh = np.floor(day_frac*24)
    t_rem = day_frac*24-hh
    mm = np.floor(t_rem*60)
    t_rem = t_rem*60-mm
    ss = t_rem*60
    frac_hh_mm_ss = np.hstack((np.hstack((np.hstack((day_frac,hh)),mm)),ss))
    step_vals = np.zeros((nsteps))
    y_steps = np.zeros(x.shape)

    if nsteps > 0:
        for i in range(nsteps):
            step_vals[i] = 2*max_step_size*(np.random.rand(1)-0.5)
            y_steps[step_locs[i]:] = y_steps[step_locs[i]:] + \
            (1 - day_frac[i])*step_vals[i]
            y_steps[(step_locs[i]+1):] = y_steps[(step_locs[i]+1):] + \
            (day_frac[i])*step_vals[i]
            step_locs_out = step_locs[0:(nsteps)]
    else:
        step_vals = np.zeros(0)
        step_locs_out = np.zeros(0)


    # Adding the arctan shapes
    slow_slip_all = np.zeros((num_arctan,x.size))
    for i in range(num_arctan):
        #tt = np.random.permutation(x.size)
        if i == 0:
            t_onset = int(x.size/2)
            slow_slip = np.arctan((x-t_onset)/(max_Darctan*np.random.rand(1)))
            slow_slip = slow_slip - np.min(slow_slip)
            slow_slip_all[i,:] = np.sign(np.random.rand(1)-0.5)*max_Aarctan*(slow_slip/np.max(slow_slip))
        if i == 1:
            t_onset = int(0.95*x.size)
            slow_slip = np.arctan((x-t_onset)/(max_Darctan*np.random.rand(1)))
            slow_slip = slow_slip - np.min(slow_slip)
            slow_slip_all[i,:] = np.sign(np.random.rand(1)-0.5)*max_Aarctan*(slow_slip/np.max(slow_slip))
        
    y_arctan = slow_slip_all.sum(axis=0)

    # Adding the Gaussian shapes
    gau_all = np.zeros((num_gau,x.size))
    for i in range(num_gau):
        tt = np.random.permutation(x.size)
        t_onset = tt[0]
        my_gau = max_Agau*np.random.rand(1)*np.exp(-((x-t_onset)**2) / \
        (2*(np.random.rand(1)*max_Dgau)**2))
        gau_all[i,:] = np.sign(np.random.rand(1)-0.5)*my_gau
    y_gau = gau_all.sum(axis=0)

    # Adding a decay at the location of the largest step
    if decay_on == 1 and nsteps > 0:
        pp = np.where(np.abs(step_vals) == np.max(abs(step_vals)))
        pp = pp[0][0]
        x_dec = np.zeros(x.size)
        x_dec[(step_locs_out[pp]):] = np.arange(1,(x_dec[(step_locs_out[pp]):]).size+1,1)-1
        y_dec = Adec*np.log10(1+x_dec/Tdec)
    else:
        y_dec = np.zeros(x.size)

    # Making the linear trend plus offset
    if mc_on == 1:
        y_sec = sec_rate*x + offset
    else:
        y_sec = np.zeros(x.size)

    # Combining the signals
    y_full = y_sec + y_seas_sto + y_steps + y_arctan + y_gau + y_dec + y_noise;

    return y_full, y_seas_sto




################################################################################################################################################################################################################################################################
def get_G_from_freqs(x,freqs_in):
    G = np.zeros([x.size,2*freqs_in.size])
    for i in range(freqs_in.size):
        G[:,2*i] = np.sin(x/(365.25/freqs_in[i])*2*np.pi)
        G[:,2*i+1] = np.cos(x/(365.25/freqs_in[i])*2*np.pi)
    return G
        
def get_y(series_in,G,tik_mul):
    y = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T,G)+tik_mul*np.eye(G.shape[1])),G.T),series_in)
    return y


def get_G_from_freqs(x,freqs_in):
    G = np.zeros([x.size,2*freqs_in.size])
    for i in range(freqs_in.size):
        G[:,2*i] = np.sin(x/(365.25/freqs_in[i])*2*np.pi)
        G[:,2*i+1] = np.cos(x/(365.25/freqs_in[i])*2*np.pi)
    return G
        
def get_y(series_in,G,tik_mul):
    y = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T,G)+tik_mul*np.eye(G.shape[1])),G.T),series_in)
    return y




################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
## Function for getting X and Y matrices from the time series
# 'a' is a two column matrix, the first column containing the time series, the second containing the targeted stochastic seasonal.
def get_XY(a,each_side):
    X_sub = np.zeros([a.shape[0]-2*each_side,2*each_side+1])
    Y_sub = np.zeros([a.shape[0]-2*each_side,2*each_side+1])
    for j in range(a.shape[0])[each_side:-each_side]:
        X_sub[j-each_side,:] = a[j-each_side:j+each_side+1,0]
        Y_sub[j-each_side,:] = a[j-each_side:j+each_side+1,1]
    ## Now taking diff and removing the nanmedian from each sample (this might not be optimal but can be experimented with later)
    
    X_diff = np.diff(X_sub,axis=1)
    X_diff = X_diff - matlib.repmat(np.nanmedian(X_diff,axis=1)[:,None],1,X_diff.shape[1])
    #X = X_sub - matlib.repmat(np.nanmedian(X_sub,axis=1)[:,None],1,X_sub.shape[1])
    
    return X_diff, Y_sub

def get_XY_fullsigs(a,each_side,G,tik_mul):
    X_sub = np.zeros([a.shape[0]-2*each_side,2*each_side+1])
    Y_sub = np.zeros([a.shape[0]-2*each_side,2*each_side+1])
    for j in range(a.shape[0])[each_side:-each_side]:
        
        series_in = a[j-each_side:j+each_side+1,0]
        m = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T,G)+tik_mul*np.eye(G.shape[1])),G.T),series_in)
        series_in = series_in - np.matmul(G,m)
        
        X_sub[j-each_side,:] = series_in
        Y_sub[j-each_side,:] = a[j-each_side:j+each_side+1,1]
        
    ## Now taking diff and removing the nanmedian from each sample (this might not be optimal but can be experimented with later)
    
    #X_diff = np.diff(X_sub,axis=1)
    #X_diff = X_diff - matlib.repmat(np.nanmedian(X_diff,axis=1)[:,None],1,X_diff.shape[1])
    
    X = X_sub - matlib.repmat(np.nanmedian(X_sub,axis=1)[:,None],1,X_sub.shape[1])
    
    
    return X, Y_sub


def get_XY_iqr(a,each_side):
    X_sub = np.zeros([a.shape[0]-2*each_side,2*each_side+1])
    Y_sub = np.zeros([a.shape[0]-2*each_side,2*each_side+1])
    for j in range(a.shape[0])[each_side:-each_side]:
        X_sub[j-each_side,:] = a[j-each_side:j+each_side+1,0]
        Y_sub[j-each_side,:] = a[j-each_side:j+each_side+1,1]
    ## Now taking diff and removing the nanmedian from each sample (this might not be optimal but can be experimented with later)
    
    X_diff = np.diff(X_sub,axis=1)
    X_diff = X_diff - matlib.repmat(np.nanmedian(X_diff,axis=1)[:,None],1,X_diff.shape[1])
    #iqrs = np.percentile(X_diff,75,axis=1) -np.percentile(X_diff,25,axis=1)
    #X_diff = np.divide(X_diff,matlib.repmat(iqrs[:,None],1,X_diff.shape[1]))
    #Y_sub = np.divide(Y_sub,matlib.repmat(iqrs[:,None],1,Y_sub.shape[1]))
    iqrs = np.percentile(X_diff.ravel(),75) -np.percentile(X_diff.ravel(),25)
    X_diff = X_diff/iqrs
    Y_sub = Y_sub/iqrs
    return X_diff, Y_sub, iqrs
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
## This function takes the output and then averages th evalues corresponding to the same time in the time series
def average_from_multi_output(a,mode=None):
    b = np.zeros([a.shape[0]+a.shape[1]-1,a.shape[1]])
    b.fill(np.nan)
    for i in range(b.shape[1]):
        b[i:i+a.shape[0],i] = a[:,i]
    
    if mode == 'median':
        out = np.nanmedian(b,axis=1)
    else:
        out = np.nanmean(b,axis=1)
    return out

def average_from_multi_output_iqr(a,iqrs,mode=None):
    #a = np.multiply(a,matlib.repmat(iqrs[:,None],1,a.shape[1]))
    a = a*iqrs
    b = np.zeros([a.shape[0]+a.shape[1]-1,a.shape[1]])
    b.fill(np.nan)
    for i in range(b.shape[1]):
        b[i:i+a.shape[0],i] = a[:,i]
    
    if mode == 'median':
        out = np.nanmedian(b,axis=1)
    else:
        out = np.nanmean(b,axis=1)
    return out



################################################################################################################################
################################################################################################################################
### For the high-pass filtering attempts
def get_non_seasonals(a,each_side,G,tik_mul):
    out = np.zeros([a.shape[0]-2*each_side,2*each_side+1])
    #print(G.shape)
    #print(out.shape)
    pp = 0
    for j in range(a.shape[0])[each_side:-each_side]:
        #print(j)
        series_in = a[j-each_side:j+each_side+1]
        #print(series_in.shape)
        m = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T,G)+tik_mul*np.eye(G.shape[1])),G.T),series_in)
        out[pp,:] = np.matmul(G[:,0:-4],m[0:-4])
        #out[pp,:] = series_in - np.matmul(G[:,0:-4],m[0:4])
        
        #out[pp,:] = np.matmul(G,m)
        #out[pp,:] = series_in - np.matmul(G,m)
        
        pp+=1
    return out
        
        
