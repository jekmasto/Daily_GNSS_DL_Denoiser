from funcs_4_DL_resids import id_names_txt,derivative,grouping,id_names_npz,Data_to_use
import random
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import matplotlib
#%matplotlib nbagg
from sklearn.metrics import mean_squared_error
import datetime as dt
import tensorflow as tf
from tensorflow import keras
import scipy.interpolate as interpolate
import scipy.signal as signal

def split_nan(x):
    """
    create groups of non nans elements
    """
    return np.split(x, np.where(np.diff(np.isnan(x), prepend=True))[0])[1::2]

def randomize_noise(r):
    '''
    Transforming the original data into the frequency domain
    randomizing the phases and converting the data back into the time domain.
    '''
    ts_fourier  = np.fft.rfft(r)
    random_phases = np.exp(np.random.uniform(0,np.pi,int(len(r)/2)+1)*1.0j)
    ts_fourier_new = ts_fourier*random_phases
    return np.fft.irfft(ts_fourier_new)

def generate_syntethics(n,soln_folder_path,save_folder_path,sec_rateA,offsetA,decay_onA,components):
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
    
    Returns
    ----------
    """
    print(n)
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
        
        y_full, y_sec, y_seas, y_seas_sto, y_steps, y_arctan, y_gau, y_dec, y_noise, \
        step_locs_out, step_vals, frac_hh_mm_ss = \
            synth_series(t,seas_freqs,seas_amp,sto_mul,max_nsteps,max_step_size,\
                        noise_level,offset,sec_rate,num_arctan,max_Aarctan,max_Darctan,\
                                   num_gau,max_Agau,max_Dgau,decay_on,Adec,Tdec,mc_on)
        step_times = step_locs_out
        step_fracs = frac_hh_mm_ss[:,0]
        #known_steps = np.hstack([step_times[:,None],step_fracs[:,None]])        
         
        ### put gaps from the residuals
        ind=np.setdiff1d(np.union1d(t, t_in), np.intersect1d(t, t_in))
        y_full[ind]=np.nan
        y_full = y_full[~np.isnan(y_full)]

        ### Putting nans in the generated time series 
        if np.isnan(t_in).any():
            nan_groups=split_nan(t_in) #nan groups
            for gr in nan_groups:
                indices=np.nonzero(np.in1d(t_in,gr))[0]
                y_full[indices]=np.nan
        
        
        assert len(y_full)==len(r_in)

        ### randomize the phase of the noise
        r_in=randomize_noise(r_in)

        ### sum generate signals and residuals [Noise]
        if len(y_full)==len(r_in):
            d_in=y_full+r_in
        else:
            d_in=y_full[1:]+r_in
            t_in=t_in[1:]

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
                stepind.append(np.argwhere(t==t.flat[np.abs(t - step_times[ii]).argmin()])[0][0]+1)
            
        targety[stepind]=1
        
        ### put nan in the targets
        for ind in np.argwhere(np.isnan(np.array(r))):
            targety[ind[0]]=np.nan
        
        out = np.vstack([t,d,r,targety]).T
        ######  Save raw data and residuals ######
        ### Save
        out_path=save_folder_path+'/'+str(i)+'.txt'
        np.savetxt(fname=out_path,X=out)
    return 

def shuffle_txt_indices(path_indexes):
    from random import shuffle
    """
    Shuffle a txt file
    
    Parameters
    ----------
       path_indexes: path of the file that you want to shuffle
       
    Returns
    ----------
       Overwrite prexisting file in a shuffled order 
    
    """
    
    lines = open(path_indexes).readlines()
    numbers=list(range(len(lines)))
    shuffle(numbers)
    lines=[lines[i] for i in numbers]
    open(path_indexes, 'w').writelines(lines)
    return numbers
    
def XY_from_d_and_length(d,r,h,input_length):
    
    """
    Create X and y with the windowing approach (X and Y can be both vectors - eg. many to many approach)
       NB: take care that it can include examples with Nan values inside
    
    Parameters
    ----------
       d: raw data vector
       r: residuals vector
       h: heaviside vector
       input_length: len of the window
    
    Returns
    ----------
        X: matrix of inputs
        Y: matrix of targets
        H_vector_zo: matrix of targets step model
    
    """
    
    X = np.zeros([d.size+input_length-1,input_length])
    X.fill(np.nan)
    
    Y = np.zeros([r.size+input_length-1,input_length])
    Y.fill(np.nan)

    H_vector = np.zeros([h.size+input_length-1,input_length])
    H_vector.fill(np.nan)
    
    for i in range(input_length):
        X[i:i+d.size,i] = d
        Y[i:i+r.size,i] = r
        H_vector[i:i+r.size,i]=h
    X = X[:,::-1]
    Y = Y[:,::-1]
    H_vector = H_vector[:,::-1]
    
    X = X[input_length-1:-input_length+1,:]
    Y = Y[input_length-1:-input_length+1,:]
    H_vector= H_vector[input_length-1:-input_length+1,:]
    
    ##### removing the median #####
    X = X - np.repeat(np.nanmedian(X,axis=1)[:,None],X.shape[1],axis=1)  
    
    return X,Y,H_vector
    
    
gen_jjj = np.vectorize(lambda x,y,z: dt.date.toordinal(dt.date(x,y,z)))

    
def Make_X_Y(max_gap,input_length,load_path,save_path,components):
    
    """
    Create X and y for one folder and N components
    
    Parameters
    ----------
       max_gap: maximum gap for having nans in the series (integer)
       input_length: length of the window to be considered as input (integer
       load_path: folder of txt files
       save_path: folder where saving
       components: components to use
    
    Returns
    ----------
        Save .npz file for each component of each station
        One .npz file contain: X=X,Y=Y,t=t,d=d,r=r
           X: input examples
           y: target examples
           H_vector: targets examples step model
           t: time vector
           d: raw data vector
           r: residuals vector 
           h: heaviside vector
    """
    if  isinstance(max_gap, float):   
        assert max_gap.is_integer(), 'max_gap should be an integer' 
    if  isinstance(input_length, float):       
        assert input_length.is_integer(), 'max_gap should be an integer'       
    
    names=id_names_txt(load_path+'/'+components[0])
    isExist = os.path.exists(save_path)
    if not isExist:
        os.mkdir(save_path)
    
    for c in range(len(components)):
        print('Component: '+components[c])
        if os.path.isdir(save_path+'/'+components[c]):
            for f in os.listdir(save_path+'/'+components[c]):
                os.remove(os.path.join(save_path+'/'+components[c], f))
        else:
            os.mkdir(save_path+'/'+components[c])
       
        save_path_component=save_path+'/'+components[c]
        load_path_component=load_path+'/'+components[c]
        
        for i in range(len(names)):
            in_path = load_path_component+names[i]+'.txt'
            tdr = np.loadtxt(in_path)
            t_in = tdr[:,0]
            d_in = tdr[:,1]
            r_in = tdr[:,2]
            h_in = tdr[:,3]
            
            t,d,r,h=t_in,d_in,r_in,h_in
        
            #### Making X and y
            X,Y,H_vector = XY_from_d_and_length(d,r,h,input_length)

            ####################################################
            ###### Create list of transients ######
            
            ## go to acceleration
            ####################################################
            ## These are arbitrary values ##
            thr=0.0002  ##threshold value to identify a transient
            dist=5 #temporal distance
            acceleration=derivative(t[1:],derivative(t,d-r))
            groups=grouping(np.abs(acceleration),t[2:],thr,dist)
            if len(groups)>0:
                list_transient=np.array([group[0]-3 for group in groups])
            else:
                list_transient=[]

            t=t[input_length-1:]

            ### Saving with nans still inside
            save_path_s = save_path+'/'+components[c]+names[i]+'.npz'       
            np.savez(file=save_path_s,X=X,Y=Y,H=H_vector,t=t,d=d,r=r,h=h,list_transient=list_transient)
    return    