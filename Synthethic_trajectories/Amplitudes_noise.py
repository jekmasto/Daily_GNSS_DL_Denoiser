# SET A DATASET
import os
import numpy as np
import matplotlib.pyplot as plt 
import random


def handle_steps(esempioI,esempiogrt,step_ex):
    
    '''
    This function masks the input tensor of 1 and 0, based on the ratioT, that is the AD/MAD of each step
    
    Parameters
    ----------
        esempioI: 'float64' 
            raw array 
        esempiogrt: 'float64' 
            denoised array 
        step_ex: 'float64' 
            array of 0 and 1
     Returns
     ----------
        amplitude: list of lists ('float64')
            list of amplitudes of each step
        mad: list of 'float64'
            list of MAD (Mean absolute deviation) of each example
        esempio: 'float64'
            corrected array
    '''
    
    
    esempio=esempiogrt.copy()
    st_locT=np.argwhere(step_ex==1)
    amplitude=[]
    input_length=esempioI.shape[0]
    
    for i in range(len(st_locT)):
        st_loc=st_locT[i][0]
        #print('position',st_loc)
        
        if st_loc  < input_length-2:
            if st_loc > 0:
                index_before_given = np.nanargmax(~np.isnan(esempio[:st_loc][::-1]))
            else:
                index_before_given=st_loc
                
            # Calculate the actual index from the end of the array
            actual_index = st_loc - index_before_given - 1
            #index_after_given = np.argmax(~np.isnan(esempio[(st_loc+1)+1:])) + (st_loc+1) + 1
            # Find the first index after the given index that contains a non-NaN value
            index_after_given = st_loc + 2
            while index_after_given < len(esempio)-1 and np.isnan(esempio[index_after_given]):
                index_after_given += 1
            #print('first no -nan index: ', index_after_given)
       
            
            if ~np.isnan(esempio[st_loc]) and index_after_given!=st_loc+2:
                if ~np.isnan(esempio[st_loc+1]):
                    esempio[st_loc+1]=esempio[st_loc+1]-(esempio[st_loc+1]-esempio[st_loc])
                esempio[index_after_given:]=esempio[index_after_given:]-(esempio[index_after_given]-esempio[st_loc])
                #print('Index step no nan but the one after yes - it works')
                actual_indexN=st_loc
            
            if np.isnan(esempio[st_loc]) and index_after_given==st_loc+2:   
                if ~np.isnan(esempio[st_loc+1]):
                    esempio[st_loc+1]=esempio[st_loc+1]-(esempio[st_loc+1]-esempio[st_loc])
                esempio[index_after_given:]=esempio[index_after_given:]-(esempio[index_after_given]-esempio[actual_index])
                #print('Index step nan but the one after no - it works')
                actual_indexN=actual_index
                
            if np.isnan(esempio[st_loc]) and index_after_given!=st_loc+2:    
                #print('Index step nan and also the one after nan')
                esempio[index_after_given:]=esempio[index_after_given:]-(esempio[index_after_given]-esempio[actual_index])
                actual_indexN=actual_index
                
            if index_after_given==st_loc+2 and ~np.isnan(esempio[st_loc]):
                if ~np.isnan(esempio[st_loc+1]):
                    esempio[st_loc+1]=esempio[st_loc+1]-(esempio[st_loc+1]-esempio[st_loc])
                    #print('No nan')
                #else:
                    #print('Nan step index+1')    
                esempio[st_loc+2:]=esempio[st_loc+2:]-(esempio[st_loc+2]-esempio[st_loc])
                actual_indexN=st_loc
             
            amplitude.append(np.abs(esempiogrt[index_after_given]-esempiogrt[actual_indexN]))
    
        else:
            amplitude.append(np.nan)
            
            
        mad=np.nanmedian(np.abs(esempio - np.nanmedian(esempio)))
            
    return amplitude,mad,esempio


def mask_thr(YTV,ratioT,indexes,thr):
    
    '''
    This function masks the input tensor of 1 and 0, based on the ratioT, that is the AD/MAD of each step
    
    Parameters
    ----------
        YTV: 'float64' 
            tensor of 0 and 1
        ratioT: list of 'float64' 
            list of ratios
        indexes: list of lists of integers
            list of indexes that contains steps
        thr: 'float64' 
            Threshold based on which you want to mask your tensor
         
    Returns
    -------
        YTV_thr: 'float64' 
            tensor of 0 and 1 that is masked
    '''
    
    YTV_thr=YTV.copy()
    for i in range(len(ratioT)):
        if len(ratioT[i])==1:
            if ratioT[i][0]<thr:
                YTV_thr[indexes[i][0],indexes[i][1][0]]=0
        else:
            higher_i=np.argwhere(np.array(ratioT[i])<thr)[:,0]
            #print(higher_i)
            for h in range(len(higher_i)):
                YTV_thr[indexes[i][0],indexes[i][1][higher_i[h]]]=0
                        
    return YTV_thr

def calc_ad_mad(esempio,st_loc):
     
     '''
     This function takes an array of 'float 64' and correct it for steps in order to calculate the MAD
     
     Parameters
     ----------
         esempio: 'float64' 
             input array to clear
         st_loc: list of 'integers' 
             list of indexes of steps
     Returns
     -------
         actual_indexN: 'integer'
             first previous index of the input array no Nan
         index_after_given: 'integer'
             first subsequent index of the input array no Nan
         esempio: 'float64' 
             corrected array
     '''     
     
     index_before_given = np.nanargmax(~np.isnan(esempio[:st_loc][::-1]))
    
     # Calculate the actual index from the end of the array
     actual_index = st_loc - index_before_given - 1
               # Find the first index after the given index that contains a non-NaN value
     index_after_given = st_loc + 2
     while index_after_given < len(esempio)-1 and np.isnan(esempio[index_after_given]):
         index_after_given += 1
     
        
     #print('first no -nan index: ', index_after_given)
    
     
     if ~np.isnan(esempio[st_loc]) and index_after_given!=st_loc+2:
         if ~np.isnan(esempio[st_loc+1]):
             esempio[st_loc+1]=esempio[st_loc+1]-(esempio[st_loc+1]-esempio[st_loc])
         esempio[index_after_given:]=esempio[index_after_given:]-(esempio[index_after_given]-esempio[st_loc])
         #print('Index step no nan but the one after yes - it works')
         actual_indexN=st_loc
     
     if np.isnan(esempio[st_loc]) and index_after_given==st_loc+2:   
         if ~np.isnan(esempio[st_loc+1]):
             esempio[st_loc+1]=esempio[st_loc+1]-(esempio[st_loc+1]-esempio[st_loc])
         esempio[index_after_given:]=esempio[index_after_given:]-(esempio[index_after_given]-esempio[actual_index])
         #print('Index step nan but the one after no - it works')
         actual_indexN=actual_index
         
     if np.isnan(esempio[st_loc]) and index_after_given!=st_loc+2:    
         #print('Index step nan and also the one after nan')
         esempio[index_after_given:]=esempio[index_after_given:]-(esempio[index_after_given]-esempio[actual_index])
         actual_indexN=actual_index
         
     if index_after_given==st_loc+2 and ~np.isnan(esempio[st_loc]):
         if ~np.isnan(esempio[st_loc+1]):
             esempio[st_loc+1]=esempio[st_loc+1]-(esempio[st_loc+1]-esempio[st_loc])
             #print('No nan')
         #else:
             #print('Nan step index+1')    
         esempio[st_loc+2:]=esempio[st_loc+2:]-(esempio[st_loc+2]-esempio[st_loc])
         actual_indexN=st_loc
    
     return actual_indexN,index_after_given,esempio


def handle_steps_whole_TS(XTv,YTvg,YTv):
    '''
    This function takes an array of 'float 64' and correct it for steps in order to calculate the MAD
     
    Parameters
    ----------
        XTv: 'float64' 
            array of raw data
        YTvg: 'float64' 
            array of noise
        YTv: 'float64' 
            array of steps
             
    Returns
    -------
        amplitudeT: list of arrays
            where each value is the amplitude of the step in the denoised array
        madT: list of MAD (Mean absolute deviation) of each example
        ratioT: list of lists
            where each element is the ratio between Amp/MAD
        indexes: list o lists (integers)
            where the first element of each list is the example of YTv while the elements of the second list are the indexes of the example 
    '''
    
    st_locT=np.array(np.argwhere(YTv==1))[:,0]
    input_length=YTv.shape[1]
    sizeTot=XTv.shape[0]
    
    amplitudeT=[]
    madT=[]
    ratioT=[]
    
    #
    indexes=[]
    
    for ii in range(len(st_locT)):
        
        i=st_locT[ii]
        esempioI=XTv[i]
        esempiogrt=XTv[i]-YTvg[i]
        step_ex=YTv[i,:]
        esempio=esempiogrt.copy()
        
        st_loc_es=np.argwhere(YTv[st_locT[ii],:]==1)[:,0]
        
        #print('position',st_loc)
        indexes.append([i,st_loc_es])
            
        amplitude=[]
        for k in range(len(st_loc_es)):
            
            st_loc=st_loc_es[k]
            
            if st_loc  > 2 and st_loc < input_length-2:
                
                index_after_given,actual_indexN,esempio=calc_ad_mad(esempio,st_loc)
                 
                amplitude.append(np.abs(esempiogrt[index_after_given]-esempiogrt[actual_indexN]))
                mad=np.nanmedian(np.abs(esempio - np.nanmedian(esempio)))
            
            elif st_loc  < 2 and st_loc < input_length-2 and i !=0:
                
                st_loc=st_loc+input_length
                #Combine the example before
                esempioI=np.concatenate([XTv[i-1], XTv[i]])
                esempiogrt=esempioI-np.concatenate([YTvg[i-1],YTvg[i]])
                step_ex=np.concatenate([YTv[i-1,:],YTv[i,:]])
                esempio=esempiogrt.copy()
                
                index_after_given,actual_indexN,esempio=calc_ad_mad(esempio,st_loc)
                amplitude.append(np.abs(esempiogrt[index_after_given]-esempiogrt[actual_indexN]))
                mad=np.nanmedian(np.abs(esempio[st_loc:st_loc+input_length] - np.nanmedian(esempio[st_loc:st_loc+input_length])))
             
            elif st_loc  > 2 and st_loc > input_length-2 and i!= sizeTot-1:
               
                #Combine the example before
                esempioI=np.concatenate([XTv[i], XTv[i+1]])
                esempiogrt=esempioI-np.concatenate([YTvg[i],YTvg[i+1]])
                step_ex=np.concatenate([YTv[i,:],YTv[i+1,:]])
                esempio=esempiogrt.copy()
                
                index_after_given,actual_indexN,esempio=calc_ad_mad(esempio,st_loc)
                amplitude.append(np.abs(esempiogrt[index_after_given]-esempiogrt[actual_indexN]))
                mad=np.nanmedian(np.abs(esempio[:input_length] - np.nanmedian(esempio[:input_length])))
            
            else:
                amplitude.append(np.nan)
                mad=np.nan
        
        ratio=np.array(amplitude)/mad
        ratio=[round(r,3) for r in ratio]     
        
        amplitudeT.append(np.array(amplitude))
        ratioT.append(ratio)
        madT.append(mad)
        
    return amplitudeT,madT,ratioT,indexes

def identify(step_examples,XTv,YTv,YTvg,thr=None):
    
    '''
    This function takes an array of 'float 64' and correct it for steps in order to calculate the MAD
     
    Parameters
    ----------
        XTv: 'float64' 
            array of raw data
        YTv: 'float64' 
            array of steps data
        YTvg: 'float64' 
            array of noise
        step_examples: 'integers' 
            'indexes of steps examples
            
        thr: 'float64'
            threshold to identify the list under
        
    Returns
    ----------
        ratioT: list of ratios
        amplitudesF: list  of amplitudes
        under: list of exaples below the threshold
        
    '''
    
    
    ratioT=[]
    amplitudesF=[]
    under=[]
    
    for i in range(len(step_examples)):
        location=step_examples[i]
        esempioI=XTv[location]
        esempiogrt=XTv[location]-YTvg[location]
        step_ex=YTv[location,:]
        
        ad,mad,esempio=handle_steps(esempioI,esempiogrt,step_ex)
        ratio=ad/mad
        ratio=[round(r,3) for r in ratio]
        ratioT.append(ratio)
        if len(ratio)>1:
            ratiof=np.array(ratio)
            for r in ratiof:
                if r<4.4478:
                    under.append(i)
        else:
            if ratio[0]<4.4478:
                under.append(i)
            
        amplitudesF.append(ad)
    
    if under is None:
        return ratioT,amplitudesF
    else:
        return ratioT,amplitudesF,under

def gutenberg_richter_law(magnitude, a, b):
    """
    Gutenberg-Richter law function.
    
    Parameters
    ---------
      magnitude: Magnitude of the earthquakes.
      a: Constant representing the total number of earthquakes
      b: Constant characterizing the slope of the distribution

    Returns
    ---------
      N: The number of earthquakes with magnitude greater than or equal to M.
    """
    return 10 ** (a - b * magnitude)


def f_scaling(XTr,XTv,scaling):
    
    """
    scaling the tensors
    """
    
    if scaling=='Mean':
        mean=np.nanmean(XTr)
        std=np.nanstd(XTr)
        mean=-2.1305276090027813e-05 
        std=0.02739616196321316

        
        print('Mean: ', mean,'Std: ',std)
             
        for jj in range(XTr.shape[0]):
            XTr[jj,:]=(XTr[jj,:]-mean)/std
            if jj<XTv.shape[0]:
                XTv[jj,:]=(XTv[jj,:]-mean)/std
    if scaling=='MaxMin':
        Min=XTr.min()
        Max=XTr.max()
        XTr=(XTr-Min)/(Max-Min)
        XTv=(XTv-Min)/(Max-Min)
    return XTr,XTv

def interpolate_nan(example):
    # Apply interpolation along the second axis for each example
    nan_indices = np.isnan(example)
    non_nan_indices = ~nan_indices
    x = np.arange(len(example))

    # Interpolate only if there are NaN values in the example
    if np.any(nan_indices):
        #example[nan_indices]=-10
        example[nan_indices] = np.interp(x[nan_indices], x[non_nan_indices], example[non_nan_indices])
    
    return example


