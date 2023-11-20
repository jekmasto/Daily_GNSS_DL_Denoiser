"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Function for generating Fake GNSS daily displacement time-series

@author: jon, giacomo
"""
import numpy as np
from numpy import matlib
import random

def gutenberg_richter_law(magnitude, a, b):
    """
    Gutenberg-Richter law function.
    
    Parameters
    ---------
      magnitude: Magnitude of the earthquake.
      a: Constant representing the total number of earthquakes
      b: Constant characterizing the slope of the distribution

    Returns
    ---------
      N: The number of earthquakes with magnitude greater than or equal to M.
    """
    return 10 ** (a - b * magnitude)

def put_outliers(data,outliers_amplitudes,window):
    
    '''
    Parameters
    --------
       data: input array
       outliers_amplitudes: amplitudes in AD/MAD of the n outliers to be added
       window: window (in samples) to consider to calculate the statistics
       
    Returns
    --------
       dataN: input array
       boolean_outliers: array of 0 and 1, where 1 means it is an outlier
    
    '''
    
    dataN=data.copy()
    boolean_outliers=np.zeros(len(dataN))
    n_outliers=len(outliers_amplitudes)
    locations_of_outliers=np.random.choice(range(int(window/2),len(data)),n_outliers)
    boolean_outliers[locations_of_outliers]=1
    
    for i in range(len(locations_of_outliers)): 
        sign=[1 if random.random() < 0.5 else -1][0]
        size=outliers_amplitudes[i]*sign
        start=locations_of_outliers[i]-int(window/2)
        end=locations_of_outliers[i]+int(window/2)
        # Compute the median absolute deviation for the window
        window_mad = np.median(np.abs(data[start:end] - np.median(data[start:end])))
        # Check for outliers
        dataN[locations_of_outliers[i]]=(size*window_mad)+np.median(data[start:end])
    
    return np.array(dataN),boolean_outliers

def introduce_nans(array, max_consecutive_nans=20, total_nan_percentage=10):
    
    '''
    Parameters
    --------
       array: input array 
       max_consecutive_nans: maximum number of admitted consecutive Nans
       total_nan_percentage: % of Nans with respect to the length of the input time series
    
    Returns
    --------
       array: with Nans inside
    
    '''
    
    length = len(array)
    max_total_nans = int(length * total_nan_percentage / 100)

    # Randomly choose the number of consecutive NaNs and their starting position
    num_consecutive_nans = min(np.random.randint(int(max_consecutive_nans/2), max_consecutive_nans + 1), max_total_nans)
    print(num_consecutive_nans)
    start_position = np.random.randint(0, length - num_consecutive_nans + 1)

    # Introduce NaN values
    array[start_position : start_position + num_consecutive_nans] = np.nan
    nan_positions = np.where(~np.isnan(array))[0]
    
    # put other random NaNs
    left_nans=max_total_nans-num_consecutive_nans
    if left_nans>0:
        # Randomly choose indices to insert NaN values
        nan_indices = np.sort(np.random.choice(nan_positions, size=left_nans, replace=False))
        array[nan_indices] = np.nan 

    return array


def synth_series(x,seas_freqs,seas_amp,sto_mul,max_nsteps,magnitude_steps,GR,noise_level,offset,sec_rate, \
    num_arctan,max_Aarctan,max_Darctan,num_gau,max_Agau,max_Dgau,decay_on,Adec,Tdec,mc_on,Outliers_flag, \
                 best_dist_percentage,best_dist_amplitudes,Nans_flag,max_consecutive_nans,total_nan_percentage):
    
    '''
    Parameters
    --------
       x: time vector (in days)
       seas_freqs: 
       seas_amp: 
       sto_mul: 
       max_nsteps:
       magnitude_steps: range of possible steps amplitudes
       weights_GR: weights of GR
       noise_level:
       offset:
       sec_rate:
       num_arctan:
       max_Aarctan: 
       max_Darctan:
       num_gau:
       max_Agau:
       max_Dgau:
       decay_on:
       Adec:
       Tdec:
       mc_on:
       Outliers_flag: boolean (if True: add outliers)
       best_dist_percentage: distribution of percentages (numbers) of outliers
       best_dist_amplitudes: distribution of amplitudes of outliers
       Nans_flag:  boolean (if True: add Nans)
       Percentage_nans: percentage of Nans to put inside
       max_consecutive_nans: 
       total_nan_percentage: None
       
    Returns
    --------
       y_full:
       y_sec:
       y_seas:
       y_seas_sto:
       y_steps:
       y_arctan:
       y_gau:
       y_dec:
       y_noise:
       step_locs_out:
       step_vals:
       frac_hh_mm_ss:
    
    '''

    # initializing y
    y = np.zeros(x.shape)
    
    ################### synthesizing the seasonal component of the signal ##################
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

    ################### Making the noise (as a fraction of the seasonal amplitude) ##################
    amp_range = np.max(y_seas)-np.min(y_seas)
    y_noise = np.random.rand(1)[0]*noise_level*amp_range*np.random.randn(x.size)

    ################### making the steps in the data ##################
    nsteps = np.round(np.random.rand(1)*max_nsteps)[0].astype(int)
    step_locs = np.random.permutation(x.size)
    y_steps = np.zeros(x.shape)
    # These are steps distributed in 2 days, to simplify the distribution is maximum 0.3/0.7
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
            #step_vals[i]= [random.uniform(-2*max_step_size,-0.005),random.uniform(0.005,2*max_step_size)][random.randrange(2)]
            #step_vals[i] = 2*max_step_size*(np.random.rand(1)-0.5)
            sign=[1 if random.random() < 0.5 else -1][0]
            step_vals[i]=np.random.choice(magnitude_steps, p=GR)*sign
            y_steps[step_locs[i]:] = y_steps[step_locs[i]:] + (1 - day_frac[i])*step_vals[i]
            y_steps[(step_locs[i]+1):] = y_steps[(step_locs[i]+1):] + (day_frac[i])*step_vals[i]
            step_locs_out = step_locs[0:(nsteps)]
    else:
        step_vals = np.zeros(0)
        step_locs_out = np.zeros(0)
    
    ################## Flag to create SSE #################
    Arctan_flag=np.random.rand(1)
    if Arctan_flag>0.5 and np.abs(sec_rate)>0.00005475814013977: #2 cm/year
        print('SSEs are created')
        y_arctan,SSE1,SSE2=plausible_SSEs(x,max_Darctan,max_Aarctan) 
    else:
        y_arctan=np.zeros(x.size)

    ######## Adding the Gaussian shapes ########
    gau_all = np.zeros((num_gau,x.size))
    for i in range(num_gau):
        tt = np.random.permutation(x.size)
        t_onset = tt[0]
        my_gau = max_Agau*np.random.rand(1)*np.exp(-((x-t_onset)**2) / \
        (2*(np.random.rand(1)*max_Dgau)**2))
        gau_all[i,:] = np.sign(np.random.rand(1)-0.5)*my_gau
    y_gau = gau_all.sum(axis=0)

    ######## Adding a decay at the location of the largest step ########
    if decay_on == 1 and nsteps > 0:
        pp = np.where(np.abs(step_vals) == np.max(abs(step_vals)))
        pp = pp[0][0]
        x_dec = np.zeros(x.size)
        x_dec[(step_locs_out[pp]):] = np.arange(1,(x_dec[(step_locs_out[pp]):]).size+1,1)-1
        y_dec = Adec*np.log10(1+x_dec/Tdec)
    else:
        y_dec = np.zeros(x.size)

    ######## Making the linear trend plus offset ########
    if mc_on == 1:
        y_sec = sec_rate*x + offset
        
        ## The linear trend should be opposite with respect to SSEs
        if y_arctan[0]>y_arctan[-1] and  sec_rate<0:
            y_sec=-y_sec
        if y_arctan[0]<y_arctan[-1]>0 and  sec_rate>0:
            y_sec=-y_sec
            
    else:
        y_sec = np.zeros(x.size)

    ######## Combining the signals ########
    y_full = y_sec + y_seas_sto + y_steps + y_arctan + y_gau + y_dec + y_noise;
    
    ######## ADD outliers ########
    if Outliers_flag==True:
        ##### Percentage of outliers #####
        params=best_dist_percentage[1]
        # Parameters for the generalized hyperbolic distribution
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        location = 0.0  # Set the location parameter if needed
        perc = best_dist_percentage[0].rvs(*arg, loc=0.0, scale=scale, size=1)
        n_outliers=int((len(y_full)/100)*perc)
        
        ##### Ampltudes of outliers #####
        params=best_dist_amplitudes[1]
        # Parameters for the generalized hyperbolic distribution
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        location = 0.0  # Set the location parameter if needed
        outliers_amplitudes= best_dist_amplitudes[0].rvs(*arg, loc=0.0, scale=scale, size=n_outliers)
        
        ## Add outliers
        N_full,boolean_outliers=put_outliers(y_full,outliers_amplitudes,window=60)
    else:
        N_full=y_full

    ######## ADD Nans ########
    if Nans_flag==True:
        if max_consecutive_nans is not None:
            if Outliers_flag ==True and total_nan_percentage is None:
                NN_full= introduce_nans(N_full,max_consecutive_nans,perc)  
            else:
                NN_full = introduce_nans(N_full,max_consecutive_nans,total_nan_percentage)        
        else:
            NN_full= introduce_nans(N_full)
            
        nan_positions = np.where(np.isnan(NN_full))[0]
        Alloutputs=[y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \
            step_locs_out, step_vals, frac_hh_mm_ss,boolean_outliers]
        for i in range(len(Alloutputs)):
            Alloutputs[i][nan_positions]=np.nan
        y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \
    step_locs_out, step_vals, frac_hh_mm_ss,boolean_outliers=Alloutputs
        
     else:
        NN_full=N_full
    
    return NN_full,y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \
    step_locs_out, step_vals, frac_hh_mm_ss,boolean_outliers


def plausible_SSEs(x,max_Darctan,max_Aarctan): 
    
    '''
    Parameters
    --------
       x: time vector (in days)
       max_Darctan:
       max_Aarctan:
       
    Returns
    --------
    
    '''
    
    ## Controls if you have a shorter cycle of SSEs
    cycles_of_SSE=np.random.rand(1)
    
    # Periodicity of arctan shapes
    percentage_changeY = random.uniform(0, 200)
    # Calculate the adjusted value for Periodicity of SSE, between 1 and 3 years
    Per_year = int(365 + (365 * (percentage_changeY / 100)))

    slow_slip_all,ampD,amplitude2=SSE_cycle(x,Per_year,max_Darctan,max_Aarctan)

    if cycles_of_SSE<0.33:
        y_arctan = slow_slip_all.sum(axis=0)
        slow_slip_allN=np.zeros(slow_slip_all.shape)
    else:
        #print('There are two cycles of SSEs')
        ampDN= ampD*random.uniform(0.05, 0.2)
        amplitude22=amplitude2*random.uniform(0.1, 0.3)
        Per_year=int(Per_year/4)
        slow_slip_allN,ampDsecond,amplitude2second=SSE_cycle(x,Per_year,max_Darctan,max_Aarctan,amplitude2=ampDN,ampD=amplitude22)
        y_arctan=slow_slip_all.sum(axis=0)+slow_slip_allN.sum(axis=0)
        
    return  y_arctan,slow_slip_all,slow_slip_allN

def SSE_cycle(x,Per_year,max_Darctan,max_Aarctan,amplitude2=None,ampD=None):
    
    '''
    Parameters
    ---------
       max_Darctan: #Controls max. width of arctan function (in time. days) 
       if amplitude2==None and ampD==None:
          then this two parameters are set randomly from max_Darctan and max_Aarctan
       else:
          this two parameters are input
       
    Returns
    ---------
       slow_slip_all: array of slow slip with size (len(SSE),len(x))
       amplitude2: time width of SSE 
       ampD: amplitude of SSE
       
    '''
    
    ## maximum number of possible SSE
    num_arctan=round(x.size/(365/4))
    
    ######### Adding the arctan shapes ########
    slow_slip_all = np.full((num_arctan,x.size), np.nan)
    
    t_onset = int(int(Per_year)* random.uniform(0.0001,0.9))
    random_amp=random.uniform(0.2,1)

    if amplitude2 is None:
        amplitude2=max_Darctan*random_amp
    
    slow_slip = np.arctan((x-t_onset)/(amplitude2))
    slow_slip = slow_slip - np.min(slow_slip)
    
    if ampD is None:
        ampD = np.sign(np.random.rand(1)-0.5)*max_Aarctan
    #print('first amplitude is: ',ampD)
    
    slow_slip=ampD*(slow_slip/np.max(slow_slip))
    slow_slipF=slow_slip

    ######## Not all SSEs have nested SSEs   ########
    #Periodic_flag = random_number = random.uniform(0, 1)
    Periodic_flag=0.1
    if Periodic_flag<0.5:
        periodicity_dec=60
        percentage_change_periodicity_dec = random.uniform(-50,50)
        # Calculate the adjusted value for Periodicity of SSE, between 1 and 6 months
        periodicity_dec_V = int(periodicity_dec + (periodicity_dec * (percentage_change_periodicity_dec / 100)))
        
        if (int((amplitude2*3)/periodicity_dec_V))>0.5:  
            t_onsetST,indeces_zero=periodic_decomposition_trend(x,t_onset,amplitude2,periodicity_dec_V)
            #print('Indices zero',indeces_zero)
            for ind in indeces_zero:
                if ind[-1]+1<slow_slip.shape[0]:
                    slow_slip[ind]=slow_slip[ind[0]-1]
            for ind in indeces_zero:
                if ind[-1]+1<slow_slip.shape[0]:
                    slow_slip[ind[-1]+1:]=slow_slip[ind[-1]+1:]-(slow_slip[ind[-1]+1]-slow_slip[ind[0]-1])
    slow_slip_all[0,:]=slow_slip
    
    ########## Create a new semi-periodic SSE ##########
    periodicityST=range(t_onset,x.size,Per_year)
    #print([iiii for iiii in range(t_onset,x.size,Per_year)])

    if len(periodicityST)>1:
        for i in range(1,len(periodicityST)):
            slow_slip=0
            percentage_change = random.uniform(-30, 30)
            t_onsetN=periodicityST[i]+ (Per_year * (percentage_change / 100))

            percentage_change = random.uniform(-30, 30)
            # Calculate the adjusted value for random_amp2
            amplitude2D = amplitude2 + (amplitude2 * (percentage_change / 100))
            
            slow_slip = np.arctan((x-t_onsetN)/(amplitude2D))
            slow_slip = slow_slip - np.min(slow_slip)
            
            ampD2 = ampD + (ampD * (percentage_change / 100))
            #print('The amplitude is ', ampD2)
            
            slow_slip=ampD2*(slow_slip/np.max(slow_slip))
            if slow_slipF[-1]>0 and slow_slip[-1]<0 :
                slow_slip = -slow_slip
            if slow_slipF[-1]<0 and slow_slip[-1]>0 :
                slow_slip = -slow_slip

            #Periodic_flag = random_number = random.uniform(0, 1)
            if Periodic_flag<0.5:
                if (int((amplitude2*3)/periodicity_dec_V))>0.5:
                    t_onsetST,indeces_zero=periodic_decomposition_trend(x,t_onsetN,amplitude2D,periodicity_dec_V)
                    for ind in indeces_zero:
                        if ind[-1]+1<slow_slip.shape[0]:
                            slow_slip[ind]=slow_slip[ind[0]-1]
                    for ind in indeces_zero:
                        if ind[-1]+1<slow_slip.shape[0]:
                            slow_slip[ind[-1]+1:]=slow_slip[ind[-1]+1:]-(slow_slip[ind[-1]+1]-slow_slip[ind[0]-1])
            slow_slip_all[i,:]=slow_slip

    slow_slip_all=slow_slip_all[:,1:]
    nan_columns = np.any(np.isnan(slow_slip_all), axis=1)
    # Remove columns with NaN values
    slow_slip_all = slow_slip_all[~nan_columns,: ]
    slow_slip_all = np.insert(slow_slip_all, 0, 0, axis=1)

    return slow_slip_all,amplitude2,ampD

def periodic_decomposition_trend(x,t_onset,amplitude,periodicity,verbose=None):
    
    '''
    Parameters
    --------
       x: time vector (in days)
       t_onset: middle time point of the large SSEs to decompose
       amplitude: time amplitude (controls the temporal duration) of the large SSE 
       periodicity: (in days),
       
    Returns
    --------
       t_onsetST: list of middle time point of each SSEs
    indeces_zero: list of middle time point of each SSEs
    
    '''
    
    ## Spacing in time between small SSE
    per=np.linspace(int(t_onset-(amplitude*1.5)),int(t_onset+(amplitude*1.5)),round((amplitude*3)/periodicity))
    
    if verbose==True:
        if len(per)>1:
            print('The first SSE cycle has nested SSEs ')
            
    indeces_zero=[]
    t_onsetST=[]
    for j in range(len(per)):
        
        ## Allows for a variation of the 25% on the periodicity of SSE
        percentage_change = random.uniform(-25, 25)
        
        # Calculate the adjusted value for random_amp2
        t_onsetS = int(per[j] + (periodicity * (percentage_change / 100)))
        
        ## The temporal duration of small SSE should be between the 15 and 30% of the large one
        ranT=int(periodicity*random.uniform(0.15, 0.35))
        indeces_zero.append(list(range(t_onsetS-ranT,t_onsetS+ranT)))
        t_onsetST.append(t_onsetS)
    
    return t_onsetST,indeces_zero


#####################################################################################
def synth_series_old(x,seas_freqs,seas_amp,sto_mul,max_nsteps,magnitude_steps,GR,noise_level,offset,sec_rate, \
    num_arctan,max_Aarctan,max_Darctan,num_gau,max_Agau,max_Dgau,decay_on,Adec,Tdec,mc_on,Outliers_flag, \
                 best_dist_percentage,best_dist_amplitudes):
    
    '''
    Parameters
    --------
       x: time vector (in days)
       seas_freqs: 
       seas_amp: 
       sto_mul: 
       max_nsteps:
       magnitude_steps: range of possible steps amplitudes
       weights_GR: weights of GR
       noise_level:
       offset:
       sec_rate:
       num_arctan:
       max_Aarctan: 
       max_Darctan:
       num_gau:
       max_Agau:
       max_Dgau:
       decay_on:
       Adec:
       Tdec:
       mc_on:
       Outliers_flag: boolean (if True: add outliers)
       best_dist_percentage: distribution of percentages (numbers) of outliers
       best_dist_amplitudes: distribution of amplitudes of outliers
       
    Returns
    --------
       y_full:
       y_sec:
       y_seas:
       y_seas_sto:
       y_steps:
       y_arctan:
       y_gau:
       y_dec:
       y_noise:
       step_locs_out:
       step_vals:
       frac_hh_mm_ss:
    
    '''

    # initializing y
    y = np.zeros(x.shape)
    
    ################### synthesizing the seasonal component of the signal ##################
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

    ################### Making the noise (as a fraction of the seasonal amplitude) ##################
    amp_range = np.max(y_seas)-np.min(y_seas)
    y_noise = np.random.rand(1)[0]*noise_level*amp_range*np.random.randn(x.size)

    ################### making the steps in the data ##################
    nsteps = np.round(np.random.rand(1)*max_nsteps)[0].astype(int)
    step_locs = np.random.permutation(x.size)
    y_steps = np.zeros(x.shape)
    # These are steps distributed in 2 days, to simplify the distribution is maximum 0.3/0.7
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
            #step_vals[i]= [random.uniform(-2*max_step_size,-0.005),random.uniform(0.005,2*max_step_size)][random.randrange(2)]
            #step_vals[i] = 2*max_step_size*(np.random.rand(1)-0.5)
            sign=[1 if random.random() < 0.5 else -1][0]
            step_vals[i]=np.random.choice(magnitude_steps, p=GR)*sign
            y_steps[step_locs[i]:] = y_steps[step_locs[i]:] + (1 - day_frac[i])*step_vals[i]
            y_steps[(step_locs[i]+1):] = y_steps[(step_locs[i]+1):] + (day_frac[i])*step_vals[i]
            step_locs_out = step_locs[0:(nsteps)]
    else:
        step_vals = np.zeros(0)
        step_locs_out = np.zeros(0)
    
    ################## Flag to create SSE #################
    Arctan_flag=np.random.rand(1)
    
    ######### Adding the arctan shapes ########
    slow_slip_all = np.full((num_arctan,x.size), np.nan)
    
    # Periodicity of arctan shapes
    percentage_changeY = random.uniform(0, 200)
    # Calculate the adjusted value for Periodicity of SSE, between 1 and 3 years
    Per_year = int(365 + (365 * (percentage_changeY / 100)))
    i=0
    slow_slip=0
    
    
    if Arctan_flag>0.5:
        print('There are SSE')
        #First SSE
        t_onset = int(int(Per_year)* random.uniform(0.0001,1))
        random_amp=np.random.rand(1)
        amplitude2=max_Darctan*random_amp
        slow_slip = np.arctan((x-t_onset)/(amplitude2))
        slow_slip = slow_slip - np.min(slow_slip)
        ampD = np.sign(np.random.rand(1)-0.5)*max_Aarctan
        slow_slip=ampD*(slow_slip/np.max(slow_slip))
        slow_slipF=slow_slip
        
        ######## Not all SSEs have nested SSEs   ########
        Periodic_flag = random_number = random.uniform(0, 1)
        slow_slipST=[0]
        
        if Periodic_flag<0.5:
            print('There are Periodic SSE')
            periodicity_dec=50
            percentage_change_periodicity_dec = random.uniform(-50,50)
            # Calculate the adjusted value for Periodicity of SSE, between 1 and 6 months
            periodicity_dec_V = int(periodicity_dec + (periodicity_dec * (percentage_change_periodicity_dec / 100)))
            slow_slipST,t_onsetST=periodic_decomposition(x,t_onset,ampD,amplitude2,periodicity_dec_V)
        
        sumS=0
        if len(slow_slipST)>1:
            sumS = np.sum(slow_slipST, axis=0)
            slow_slipEV=slow_slip-sumS
            slow_slip_all[0,:]=slow_slipEV
        else:
            slow_slipEV=slow_slip-0
            slow_slip_all[0,:]=slow_slip
        
        periodicityST=range(t_onset,x.size,Per_year)
    
        if len(periodicityST)>1:
            for i in range(1,len(periodicityST)):
                slow_slip=0
                percentage_change = random.uniform(-30, 30)
                t_onsetN=periodicityST[i]+ (Per_year * (percentage_change / 100))

                percentage_change = random.uniform(-30, 30)
                # Calculate the adjusted value for random_amp2
                random_amp2 = random_amp + (random_amp * (percentage_change / 100))
                print(random_amp2)

                amplitude2=max_Darctan*random_amp2
                slow_slip = np.arctan((x-t_onsetN)/(amplitude2))
                slow_slip = slow_slip - np.min(slow_slip)
                ampD2 = max_Aarctan
                slow_slip=ampD2*(slow_slip/np.max(slow_slip))
                if slow_slipF[-1]>0 and slow_slip[-1]<0 :
                    slow_slip = -slow_slip
                if slow_slipF[-1]<0 and slow_slip[-1]>0 :
                    slow_slip = -slow_slip

                #Periodic_flag = random_number = random.uniform(0, 1)
                slow_slipST=[0]
                if Periodic_flag<0.5:
                    slow_slipST,t_onsetST=periodic_decomposition(x,t_onsetN,ampD,amplitude2,periodicity_dec_V)
                    sumS=0
                    if len(slow_slipST)>1:
                        sumS = np.sum(slow_slipST, axis=0)
                        slow_slipEV=slow_slip-sumS
                    else:
                        slow_slipEV=slow_slip-0

                    slow_slip_all[i,:]=slow_slipEV
                else:
                    slow_slip_all[i,:]=slow_slip
                    
    slow_slip_all=slow_slip_all[:,1:]
    nan_columns = np.any(np.isnan(slow_slip_all), axis=1)
    # Remove columns with NaN values
    slow_slip_all = slow_slip_all[~nan_columns,: ]
    slow_slip_all = np.insert(slow_slip_all, 0, 0, axis=1)
    y_arctan = slow_slip_all.sum(axis=0)

    ######## Adding the Gaussian shapes ########
    gau_all = np.zeros((num_gau,x.size))
    for i in range(num_gau):
        tt = np.random.permutation(x.size)
        t_onset = tt[0]
        my_gau = max_Agau*np.random.rand(1)*np.exp(-((x-t_onset)**2) / \
        (2*(np.random.rand(1)*max_Dgau)**2))
        gau_all[i,:] = np.sign(np.random.rand(1)-0.5)*my_gau
    y_gau = gau_all.sum(axis=0)

    ######## Adding a decay at the location of the largest step ########
    if decay_on == 1 and nsteps > 0:
        pp = np.where(np.abs(step_vals) == np.max(abs(step_vals)))
        pp = pp[0][0]
        x_dec = np.zeros(x.size)
        x_dec[(step_locs_out[pp]):] = np.arange(1,(x_dec[(step_locs_out[pp]):]).size+1,1)-1
        y_dec = Adec*np.log10(1+x_dec/Tdec)
    else:
        y_dec = np.zeros(x.size)

    ######## Making the linear trend plus offset ########
    if mc_on == 1:
        y_sec = sec_rate*x + offset
        
        ## The linear trend should be opposite with respect to SSEs
        if y_arctan[-1]<0 and  sec_rate<0 or y_arctan[-1]>0 and sec_rate>0:
            y_sec=-y_sec
        else: 
            y_sec=-y_sec
    else:
        y_sec = np.zeros(x.size)

    ######## Combining the signals ########
    y_full = y_sec + y_seas_sto + y_steps + y_arctan + y_gau + y_dec + y_noise;
    
    ######## ADD outliers ########
    if Outliers_flag==True:
        ##### Percentage of outliers #####
        params=best_dist_percentage[1]
        # Parameters for the generalized hyperbolic distribution
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        location = 0.0  # Set the location parameter if needed
        perc = best_dist_percentage[0].rvs(*arg, loc=0.0, scale=scale, size=1)
        n_outliers=int((len(y_full)/100)*perc)
        
        ##### Ampltudes of outliers #####
        params=best_dist_amplitudes[1]
        # Parameters for the generalized hyperbolic distribution
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        location = 0.0  # Set the location parameter if needed
        outliers_amplitudes= best_dist_amplitudes[0].rvs(*arg, loc=0.0, scale=scale, size=n_outliers)
        
        ## Add outliers
        N_full,boolean_outliers=put_outliers(y_full,outliers_amplitudes,window=60)
    else:
        N_full=y_full
    
    return N_full,y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \
    step_locs_out, step_vals, frac_hh_mm_ss,boolean_outliers

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
def get_XY(a,each_side):
    """
    Function for getting X and Y matrices from the time series
    # 'a' is a two column matrix, the first column containing the time series, the second containing the targeted stochastic seasonal
    """
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

def average_from_multi_output(a,mode=None):
    
    """
    This function takes the output and then averages th evalues corresponding to the same time in the time series
    """
    
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
        

