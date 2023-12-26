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

def identify(step_examples,XTv,YTvg,thr=None):
    
    '''
    This function takes an array of 'float 64' and correct it for steps in order to calculate the MAD
     
    Parameters
    ----------
        XTv: 'float64' 
            array of raw data
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
    ratioT_s=[]
    amplitudesF=[]
    amplitude_sT=[]
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
#%%

input_length=61
cd='/Users/giacomomastella/Documents/PhD_giacomo/Denoiser/'+str(input_length)
os.chdir(cd)

datasets=['TRAINING','VALIDATION','TESTING']
XTv=np.load(cd+'/'+datasets[1]+'_tensor_step.npz')['X'] 
YTv=np.load(cd+'/'+datasets[1]+'_tensor_step.npz')['H'] 
YTvg=np.load(cd+'/'+datasets[1]+'_tensor_gratsid.npz')['Y']

### Take step examples
step_examples=np.array(np.argwhere(YTv==1))[:,0]
    
if len(step_examples)==0:
    raise ValueError("No steps in the input tensor.")
    
#%% Check hust one example

%matplotlib qt

step_examples=np.array(np.argwhere(YTv==1))[:,0]
random_station = random.sample(range(len(step_examples)), 1)[0]
random_station= step_examples[random_station]

i=random_station
esempioI=XTv[i]
esempiogrt=XTv[i]-YTvg[i]
step_ex=YTv[i,:]

ad,mad,esempio=handle_steps(esempioI,esempiogrt,step_ex)
ad=[round(a,3) for a in ad]
ratio=ad/mad
ratio=[round(r,3) for r in ratio]
fig, ax1 = plt.subplots(1)

# Plotting on the primary y-axis (left)
ax1.plot(esempioI,label='raw',color='blue')
ax1.plot(esempiogrt,'-.',label='trajectory',color='orange')
ax1.set_xlabel('Time')
ax1.set_ylabel('Displacement', color='blue')
ax1.plot(esempio,c='g',label='corrected')

# Creating a twin axis on the right side
axo = ax1.twinx()
axo.plot(step_ex,color='r')
axo.set_ylabel('Steps', color='r')
axo.set_title('AD/MAD ='+str(ratio)+' - amplitude='+str(ad))
ax1.legend()

plt.show()
print('Amplitude',ad)
#%%
    
ratioT,amplitudesF,under=identify(step_examples,XTv,YTvg,thr=4.4478)

#%%  
# Set constants for the GR law
a = 2.0
b = 1.0
min_step_size, max_step_size=0.005,0.1 # Maximum and minimum step size (e.g Metres) 

magnitude_steps = np.linspace(min_step_size, max_step_size, 10000)
n_values = gutenberg_richter_law(magnitude_steps, a, b)
GR = n_values / np.sum(n_values)

   
amplitudesA = np.array([item for sublist in amplitudesF for item in sublist])
ratioA = np.array([item for sublist in ratioT for item in sublist])
      
fig, axs = plt.subplots(2,2, tight_layout=True,figsize=(15,10))

a = 4.25
b = 1.0
magnitude_steps=np.linspace(min(amplitudesA),0.10,len(amplitudesA))
n_values = gutenberg_richter_law(magnitude_steps, a, b)
axs[0,0].plot(magnitude_steps,n_values,color='r')
axs[0,0].text(0.06,max(n_values)-1000,'b value =1 ',color='r')
axs[0,0].hist(amplitudesA,bins=np.linspace(0,0.15,50), color='c')
axs[0,0].set_title('Amplitudes Total [m]')

### these are thresholds for visualization
valid_indices = (np.isfinite(ratioA) & np.isfinite(amplitudesA)) & (ratioA < 20) & (amplitudesA < 0.15)
filtered_ratioT = np.array(ratioA)[valid_indices]
filtered_amplitudesF = np.array(amplitudesA)[valid_indices]

hist = axs[0,1].hist2d(filtered_ratioT, filtered_amplitudesF, bins=50, cmap='viridis')
plt.colorbar(hist[3], ax=axs[0,1], label='Counts')  # Add a color bar for reference
axs[0,1].set_title('2D Histogram: Amplitudes Tot vs AD/MAD')

axs[0,1].axvline(x=4.4478, color='r', label='Outlier_threshold')
axs[0,1].set_ylabel('Amplitudes [mm]')
axs[0,1].set_xlabel('AD/MAD')  
axs[0,1].text(4.55,0.06,'threshold=4.4478',color='r')


axs[1,0].hist(ratioA,bins=np.linspace(0,100,100), color='c')
axs[1,0].set_title('AD/MAD steps')
axs[1,0].axvline(x=4.4478, color='r', label='Outlier_threshold')

 
axs[1,1].set_title('Scatter plot: Amplitudes Tot vs AD/MAD') 
axs[1,1].scatter(ratioA,amplitudesA, s=1,color='b')
axs[1,1].axvline(x=4.4478, color='r', label='Outlier_threshold')
axs[1,1].set_ylabel('Amplitudes [m]')
axs[1,1].set_xlabel('AD/MAD')  

    
#%%
random_station = random.sample(range(len(under)), 1)[0]
random_station=under[random_station]
i= step_examples[random_station]
esempioI=XTv[i]
esempiogrt=XTv[i]-YTvg[i]
step_ex=YTv[i,:]
ad,mad,esempio=handle_steps(esempioI,esempiogrt,step_ex)
ad=[round(a,3) for a in ad]
ratio=ad/mad
ratio=[round(r,3) for r in ratio]
fig, ax1 = plt.subplots(1)

# Plotting on the primary y-axis (left)
ax1.plot(esempioI,'.',label='raw',color='blue')
ax1.plot(esempiogrt,'-',label='trajectory',color='orange')
ax1.set_xlabel('Time')
ax1.set_ylabel('Displacement', color='blue')
ax1.plot(esempio,c='g',label='corrected')
# Creating a twin axis on the right side
axo = ax1.twinx()
axo.plot(step_ex,color='r')
axo.set_ylabel('Steps', color='r')
axo.set_title('AD/MAD ='+str(ratio)+' - amplitude='+str(ad))
ax1.legend()
ax1.set_xlim([0,input_length-1])

plt.show()
print('Amplitude',ad)

#%% control the GR

step_vals=[]
for i in range(10000):
    sign = np.random.choice([-1, 1])
    magnitude = np.random.choice(magnitude_steps, p=GR)
    step_vals.append(magnitude * sign)

fig, ax1 = plt.subplots(3)
ax1[0].plot(magnitude_steps,n_values)
ax1[1].plot(GR)
ax1[2].hist(step_vals,100)

#%%

thr=3

amplitudeT,madT,ratioT,indexes=handle_steps_whole_TS(XTv,YTvg,YTv) 
assert len(np.array([item for sublist in amplitudeT for item in sublist]))==len(np.array([item for sublist in ratioT for item in sublist]))
YTV_thr= mask_thr(YTv,ratioT,indexes,thr)
            

  