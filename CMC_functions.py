#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  15 16:29:18 2023

@author: giacomo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime, math,sys
from datetime import datetime
from scipy import interpolate

def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
       origin : tuple of float (lat, long)
       destination : tuple of float (lat, long)

    Returns
    -------
       distance_in_km : float

    Examples
    --------
       origin = (48.1372, 11.5756)  # Munich
       destination = (52.5186, 13.4083)  # Berlin
       round(distance(origin, destination), 1)
       504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def weighted_median(values, weights):
    """
    Compute a weighted median based on wheights

    Parameters
    ----------
    """
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]

def max_consecutive_nan(arr):

    """
    Returns the maximum number of consecutive nans
    """

    max_ = 0
    idx = 0
    while idx < arr.size:
        while idx < arr.size and math.isnan(arr[idx]): 
            max_ += 1
            idx  += 1
        while idx < arr.size - max_:
            idx2 = idx + max_
            while idx2>idx and math.isnan(arr[idx2]):
                idx2 -=1
            if idx2==idx: 
              idx = idx + max_ +1
              break 
            idx=idx2 
        else : return max_         
    return max_ 



######## Coordinates file
file_coordinates='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/Stations_coordinates.txt'
df = pd.read_csv(file_coordinates, delimiter=',',names=['station','latitude','longitude','altitude'],header=None)
cd ='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/' # folder where you have the txt file

def build_correlation_matrix(df,cd,new_cols):
    """
    Parameters
    ----------
        df: file of coordinates with columns=['station','latitude','longitude','altitude']
        cd: folder where you have the txt file
        new_cols: names of the columns of a txt file
    """
    if not any('Gratsid' in element for element in new_cols):
        raise ValueError("The string 'Gratsid' is not present in the input columns")

    ten_step=5
    n=len(df)
    corr=np.zeros([len(df),len(df)-1,5]) 

    for i in range(len(df)):
        ### Load the dataframe of the reference station
        dfs = pd.read_csv(soln_folder_path+'/'+str(df.station[i])+'.txt', 
                 delim_whitespace=True,header=0,on_bad_lines='skip')
        new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
        dfs.rename(new_names_map, axis=1, inplace=True)
        t=list(dfs.YYMMDD)
        
        df_new=df[df.station!=df.station[i]]  
        for st in range(len(df_new)): 
            dfS = pd.read_csv(soln_folder_path+'/'+str(df_new.station[st])+'.txt', 
                 delim_whitespace=True,header=0,on_bad_lines='skip')
            new_names_map = {dfS.columns[i]:new_cols[i] for i in range(len(new_cols))}
            dfS.rename(new_names_map, axis=1, inplace=True)  
            ts=list(dfS.YYMMDD)
            ### Intersections in time
            indici1=np.nonzero(np.in1d(ts, t))[0]

            # If there is at least one day in common
            if len(indici1)>0:
                indici2=np.nonzero(np.in1d(t, ts))[0]
                dis=distance([df_new.latitude.iloc[st],df_new.longitude.iloc[st]], [df.latitude.iloc[i],df.longitude.iloc[i]])
                corr[i,st,0]=dis
                corr[i,st,1]=len(indici1)
                for c in range(len(components)):
                    r=dfs[components[c]]-dfs['Gratsid_'+components[c]]
                    rs=dfS[components[c]]-dfS['Gratsid_'+components[c]]
                    cc=np.corrcoef(rs[indici1],r[indici2])
                    corr[i,st,2+c]=cc[0,1]
        
        if (i/n)*100 > ten_step:
            print(str(ten_step)+'%')
            ten_step+=5
    return corr

def CMF(comp,t,d,r,df,station,thr_distance,cd_base,cd_data,weight_flag=None):

    """
    Return the CMC residuals
    
    Parameters
    ----------
       comp: component
       t: time vector
       d: raw data
       r: original residual
       df: all stations employed
       thr_distance: distance threshold
       cd_base: folder of the correlation matrix
       cd_data: folder of the data
       weight_flag: if True the median is weighted 

    Returns
    ----------
       t,r,d,median_res/median_resW
    """

    corr=np.load(cd_base+'Corr_'+comp+'.npy')

    # Index of the station
    ii=df.index[df.station == station].tolist()[0]
    df_new=df[df.station!=df.station[ii]] 

    ##### Correlation has to be positive #####
    indiciS=np.where((corr[ii,:,1]>0.1))[0]
    corre_ath=corr[ii,indiciS,1] #correlation
    len_ath=corr[ii,indiciS,2] #intersect_length
    distance_ath=corr[ii,indiciS,0] #distance
    df_new=df_new.iloc[indiciS] #stations to look at

    ##### Distance has to be positive #####
    above_th=np.where(distance_ath<thr_distance)[0]
    corre_ath=corre_ath[above_th]
    len_ath=len_ath[above_th]
    distance_ath=distance_ath[above_th]
    df_new=df_new.iloc[above_th]  
    
    ##### if no it means CMF cannot be applyed - the station is isolated
    if len(len_ath)!=0:
        ##### I normalize intersect_length - distance
        len_athN=(len_ath - min(len_ath)) / (max(len_ath) - min(len_ath))
        distance_athN=(distance_ath - min(distance_ath)) / (max(distance_ath) - min(distance_ath))

        ##### Here I allocate the residual of the other stations
        matrix_median=np.zeros([len(df_new),len(t)])
        n=len(df_new)
        ten_step=5

        ##### I Start the loop 
        for st in range(len(df_new)): 
            ts=np.loadtxt(cd_data+comp+'/'+df_new.station.iloc[st]+'.txt')[:,0]
            indici1=np.nonzero(np.in1d(ts, t))[0]
            if len(indici1)>0:
                indici2=np.nonzero(np.in1d(t, ts))[0]
                rs=np.loadtxt(cd_data+comp+'/'+df_new.station.iloc[st]+'.txt')[:,2]
                matrix_median[st,indici2]=rs[indici1]

            '''    
            if (st/n)*100 > ten_step:
                print(str(ten_step)+'%')
                ten_step+=5
            '''
        ##### the wheights are based on the temporal length of 1 time series and its distance
        combined_weights=len_athN*distance_athN
        median_res=np.zeros([len(t)])
        median_resW=np.zeros([len(t)])

        ###### compute the median
        for kk in range(matrix_median.shape[1]):
            indici_zero=np.where(matrix_median[:,kk]!=0)[0]
            median_res[kk]=np.nanmedian(matrix_median[indici_zero,kk])  
            if len(indici_zero)==0:  
                median_resW[kk]=np.nan
            else:
                median_resW[kk]=weighted_median(matrix_median[indici_zero,kk], combined_weights[indici_zero])
    
        ###### if there are some nans, then interpolate
        if np.isnan(median_res).any():
            print(max_consecutive_nan(median_res))
            indexes=[ind[0] for ind in np.argwhere(~np.isnan(median_res))] 
            f = interpolate.interp1d(t[indexes], median_res[indexes],fill_value="extrapolate")
            median_res = f(t)
            fW = interpolate.interp1d(t[indexes], median_resW[indexes],fill_value="extrapolate")
            median_resW = fW(t)

        if weight_flag==True:   
            return t,r,d,median_resW
        else:
            return t,r,d,median_res
    else:
        return t,r,d,0

def plot_CMC(t,r,d,median_res,comp,station):


def denoise(comp,t,d,r,df,station,thr_distance,cd_base,cd_data,weight_flag=None):

    """
    Return the CMC residuals
    
    Parameters
    ----------
       comp: component
       t: time vector
       d: raw data
       r: original residual
       df: all stations employed
       thr_distance: distance threshold
       cd_base: folder of the correlation matrix
       cd_data: folder of the data
       weight_flag: if True the median is weighted 

    Returns
    ----------
       t,r,d,median_res/median_resW
    """

    corr=np.load(cd_base+'Corr_'+comp+'.npy')

    # Index of the station
    ii=df.index[df.station == station].tolist()[0]
    df_new=df[df.station!=df.station[ii]] 

    ##### Correlation has to be positive #####
    indiciS=np.where((corr[ii,:,1]>0.1))[0]
    corre_ath=corr[ii,indiciS,1] #correlation
    len_ath=corr[ii,indiciS,2] #intersect_length
    distance_ath=corr[ii,indiciS,0] #distance
    df_new=df_new.iloc[indiciS] #stations to look at

    ##### Distance has to be positive #####
    above_th=np.where(distance_ath<thr_distance)[0]
    corre_ath=corre_ath[above_th]
    len_ath=len_ath[above_th]
    distance_ath=distance_ath[above_th]
    df_new=df_new.iloc[above_th]  
    
    ##### if no it means CMF cannot be applyed - the station is isolated
    if len(len_ath)!=0:
        ##### I normalize intersect_length - distance
        len_athN=(len_ath - min(len_ath)) / (max(len_ath) - min(len_ath))
        distance_athN=(distance_ath - min(distance_ath)) / (max(distance_ath) - min(distance_ath))

        ##### Here I allocate the residual of the other stations
        matrix_median=np.zeros([len(df_new),len(t)])
        n=len(df_new)
        ten_step=5

        ##### I Start the loop 
        for st in range(len(df_new)): 
            ts=np.loadtxt(cd_data+comp+'/'+df_new.station.iloc[st]+'.txt')[:,0]
            indici1=np.nonzero(np.in1d(ts, t))[0]
            if len(indici1)>0:
                indici2=np.nonzero(np.in1d(t, ts))[0]
                rs=np.loadtxt(cd_data+comp+'/'+df_new.station.iloc[st]+'.txt')[:,2]
                matrix_median[st,indici2]=rs[indici1]

            '''    
            if (st/n)*100 > ten_step:
                print(str(ten_step)+'%')
                ten_step+=5
            '''
        ##### the wheights are based on the temporal length of 1 time series and its distance
        combined_weights=len_athN*distance_athN
        median_res=np.zeros([len(t)])
        median_resW=np.zeros([len(t)])

        ###### compute the median
        for kk in range(matrix_median.shape[1]):
            indici_zero=np.where(matrix_median[:,kk]!=0)[0]
            median_res[kk]=np.nanmedian(matrix_median[indici_zero,kk])  
            if len(indici_zero)==0:  
                median_resW[kk]=np.nan
            else:
                median_resW[kk]=weighted_median(matrix_median[indici_zero,kk], combined_weights[indici_zero])
    
        ###### if there are some nans, then interpolate
        if np.isnan(median_res).any():
            print(max_consecutive_nan(median_res))
            indexes=[ind[0] for ind in np.argwhere(~np.isnan(median_res))] 
            f = interpolate.interp1d(t[indexes], median_res[indexes],fill_value="extrapolate")
            median_res = f(t)
            fW = interpolate.interp1d(t[indexes], median_resW[indexes],fill_value="extrapolate")
            median_resW = fW(t)

        if weight_flag==True:   
            return t,r,d,median_resW
        else:
            return t,r,d,median_res
    else:
        return t,r,d,0

def plot_CMC(t,r,d,median_res,comp,station):
    
    ### Remove the trend
    from scipy.stats import linregress
    trend=linregress(t,d)
    trend_vector=t*trend.slope+trend.intercept

    fig,axes=plt.subplots(3,2,figsize=(10,8))
    fig.subplots_adjust(wspace=.2,hspace=0.32)
    c=0

    axes[0,0].scatter(t,r,s=10,facecolor='c',linewidth=0.1,edgecolors='k')
    axes[0,0].set_title('Original Residuals Time Series  - '+comp+' - '+station)
    axes[0,1].set_title('Original Time series - '+comp+' - '+station)
    axes[0,1].scatter(t,d-trend_vector,facecolor='c',s=10,linewidth=0.1,edgecolors='k')

    axes[1,0].scatter(t,r,s=10,facecolor='c',linewidth=0.1,edgecolors='k')
    axes[1,0].set_title('Comparison  - '+comp+' - '+station)
    axes[1,0].scatter(t,r-median_res,s=5,facecolor='salmon',linewidth=0.1,edgecolors='k')
    axes[0,0].text(.5,.04,r'$\mu$ ='+str(round(np.mean(r),8))+' $\pm$ '+str(round(np.std(r),4))+'$\sigma$',horizontalalignment='center',transform= axes[0,0].transAxes,fontsize=8,
    bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'),color='b')

    axes[2,0].set_title('Denoised Residuals Time series - '+comp+' - '+station,color='salmon')
    axes[2,0].scatter(t,r-median_res,s=10,facecolor='salmon',linewidth=0.1,edgecolors='k')
    axes[2,0].text(.5,.04,r'$\mu$ ='+str(round(np.mean(r-median_res),8))+' $\pm$ '+str(round(np.std(r-median_res),4))+'$\sigma$',horizontalalignment='center',transform= axes[2,0].transAxes,fontsize=8,
    bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'),color='r')

    axes[1,1].set_title('Comparison - '+comp+' - '+station)
    axes[1,1].scatter(t,d-trend_vector,facecolor='c',s=10,linewidth=0.1,edgecolors='k')
    axes[1,1].scatter(t,d-median_res-trend_vector,s=5,facecolor='salmon',linewidth=0.1,edgecolors='k')

    axes[2,1].set_title('Denoised Time series - '+comp+' - '+station,color='salmon')
    axes[2,1].scatter(t,d-median_res-trend_vector,s=10,facecolor='salmon',linewidth=0.1,edgecolors='k')

    axes[0,1].text(.5,.04,r'RMSE ='+str(round(np.square(np.mean((r)**2)),15)),horizontalalignment='center',transform= axes[0,1].transAxes,fontsize=8,
    bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'),color='b')
    axes[2,1].text(.5,.04,r'RMSE ='+str(round(np.square(np.mean((r-median_res)**2)),15)),horizontalalignment='center',transform= axes[2,1].transAxes,fontsize=8,
    bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'),color='salmon')

    for k in range(3):
        axes[k,0].set_ylabel('Displacement [mm]',fontsize=8)
        if k<2:
            axes[2,k].set_xlabel('Time [days]')
        for j in range(2):
            axes[k,j].tick_params(axis='both', which='major', labelsize=7)
            axes[k,j].set_xlim([t[0],t[-1]])
            if k<2:
                axes[k,j].set_xticklabels([])

    if save_flag==True:
        fig.savefig(cd_save+'CMC_'+str(comp)+'_'+station+'.pdf')    
        fig.savefig(cd_save+'CMC_'+str(comp)+'_'+station+'.png') 
    
    return plt.show()

"""
#### Import correlation residuals
comp='U'
cd_base='/home/giacomo/Documents/Denoiser_GPS/Common_mode_analysis/'
cd_data='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/t_disps_resids/'
#import list of all stations
df = pd.read_csv('/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/Stations_coordinates.txt', delimiter=',',names=['station','latitude','longitude','altitude'],header=None)
station='CABU'
thr_distance=7000
save_flag=False
cd_save='/home/giacomo/Documents/Denoiser_GPS/Common_mode_analysis/Figures/pdf/'

##### Grab time series of the station to denoise
t=np.loadtxt( cd_data+comp+'/'+station+'.txt')[:,0]
d=np.loadtxt( cd_data+comp+'/'+station+'.txt')[:,1]
r=np.loadtxt( cd_data+comp+'/'+station+'.txt')[:,2]

t,r,d,median_res=denoise(comp,t,d,r,df,station,thr_distance,cd_base,cd_data)
plot_CMC(t,r,d,median_res,comp,station)
"""
