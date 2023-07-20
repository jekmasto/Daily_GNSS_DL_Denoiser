#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb  15 16:29:18 2023

Functions to apply a Common Mode Filter for GNSS daily time-series 

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

    Example
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
        values: input array
        weights: wheights of the input array
    Returns 
    ----------
        wa: weighted median
    """
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    wa=values[i[np.searchsorted(c, 0.5 * c[-1])]]
    return np.array(wa)

def check_rows(file,new_cols):
    """
    Function to skip the first lines if they do not have the proper number of columns
    """
    delimiter = ' '  # Delimiter used in the file
        
    with open(file) as f:
        for i, line in enumerate(f):
            if len(str(line).split(delimiter)) == len(new_cols) and  '' not in str(line).split(delimiter):
                break
    return np.arange(i)

def CMF(file,df,soln_folder_path,thr_distance,new_cols,Reference,Distance_file=None,save_flag=False,save_folder=None):

    """
    Return the CMC residuals
    
    Parameters
    ----------
        file: reference station you want to denoise (txt file)
        df: file of coordinates with columns=['station','latitude','longitude','altitude']
        soln_folder_path: foler whete all other txt files are 
        thr_distance: distance threshold
        new_cols: names of the columns of a txt file
        Reference: string - name of the trajectory you would like to use ('E.g. GrAtSiD or DL')
        Distance_file: list of lists, where each list includes all stations at a distance shorter than a thresholds
        save_flag: bolean (if True save the list)
        save_cd: cd where to save
       
    Returns
    ----------
        dataframe with CMF results
    """

    if not any('GrAtSiD' in element for element in new_cols) or not any('DL' in element for element in new_cols):
        raise ValueError("The string 'GrAtSiD' or 'DL' is not present in the input columns")
    if not any('E' or 'U' or 'N' in element for element in new_cols):
        raise ValueError("At least one component has to be present in the input columns")
    if not any('station' in element for element in list(df.columns)):
        raise ValueError("The string 'station' is not present in the input columns in the file of the stations coordinates ")
        
    ### Load the dataframe of the reference station
    dfs = pd.read_csv(file,delim_whitespace=True,header=0,on_bad_lines='skip', skiprows=check_rows(file,new_cols))
    data=dfs.values
    new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
    dfs.rename(new_names_map, axis=1, inplace=True)
    t=list(dfs.YYMMDD)
    
    station=file.split('/')[-1].split('.txt')[0]
    index = df.loc[df['station'] == station].index
    latitude=df.latitude.iloc[index]
    longitude=df.longitude.iloc[index]

    # Index of the station
    ii=df.index[df.station == station].tolist()[0]
    df_new=df[df.station!=df.station[ii]] 
    stations=id_names_txt(soln_folder_path)
    df_new = df_new[df_new['station'].isin(stations)]
    
    Distance_list=[]
    len_ath=[]
    indiciR=[]
    indiciS=[]
    if Distance_file is not None:
        if os.path.isfile(Distance_file):
            print("Import Distance file")
            with open("Distance_file", "rb") as fp:  
                Distance_list = pickle.load(fp)
        
        stations_to_use=np.array(Distance_list)[:,0]
        for st in range(len(stations_to_use)): 
            file_I=soln_folder_path+'/'+str(df_new.station.iloc[st])+'.txt'
            dfS = pd.read_csv(file_I, delim_whitespace=True,header=0,on_bad_lines='skip', skiprows=check_rows(file_I,new_cols))
            new_names_map = {dfS.columns[i]:new_cols[i] for i in range(len(new_cols))}
            dfS.rename(new_names_map, axis=1, inplace=True)  
            ts=list(dfS.YYMMDD)
            ### Intersections in time
            indici1=np.nonzero(np.in1d(ts, t))[0]
            indici2=np.nonzero(np.in1d(t, ts))[0]
            indiciR.append(indici1)
            indiciS.append(indici2)
            
    else:
        print("Calculate distance")
        for st in range(len(df_new)): 
            dis=distance([df_new.latitude.iloc[st],df_new.longitude.iloc[st]], [latitude,longitude])
            if dis< thr_distance:
                Distance_list.append([df_new.station.iloc[st],dis])
                file_I=soln_folder_path+'/'+str(df_new.station.iloc[st])+'.txt'
                dfS = pd.read_csv(file_I, delim_whitespace=True,header=0,on_bad_lines='skip', skiprows=check_rows(file_I,new_cols))
      
                #print(len(dfS.columns))
                new_names_map = {dfS.columns[i]:new_cols[i] for i in range(len(new_cols))}
                dfS.rename(new_names_map, axis=1, inplace=True)  
                ts=list(dfS.YYMMDD)
                ### Intersections in time
                indici1=np.nonzero(np.in1d(ts, t))[0]
                indici2=np.nonzero(np.in1d(t, ts))[0]
                indiciR.append(indici1)
                indiciS.append(indici2)

    distance_ath=np.array(Distance_list)[:,1].astype('float')
    stations_to_use=np.array(Distance_list)[:,0]
    len_ath=np.array([len(ind) for ind in indiciR])

    ##### if no it means CMF cannot be applyed - the station is isolated
    if len(Distance_list)!=0:
        ##### I normalize intersect_length - distance
        len_athN=(len_ath - min(len_ath)) / (max(len_ath) - min(len_ath))
        distance_athN=(distance_ath - min(distance_ath)) / (max(distance_ath) - min(distance_ath))

        ##### Here I allocate the residual of the other stations
        matrix_median=np.zeros([len(stations_to_use),len(t),len(components)])
        n=len(stations_to_use)
        ten_step=10

        ##### I Start the loop 
        print("Take residuals of close stations")
        for st in range(len(stations_to_use)): 
            file_I=soln_folder_path+'/'+str(df_new.station.iloc[st])+'.txt'
            dfS = pd.read_csv(file_I,delim_whitespace=True,header=0,on_bad_lines='skip', skiprows=check_rows(file_I,new_cols))
            new_names_map = {dfS.columns[i]:new_cols[i] for i in range(len(new_cols))}
            dfS.rename(new_names_map, axis=1, inplace=True)  
            ts=list(dfS.YYMMDD)
            
            indici1=indiciR[st]
            indici2=indiciS[st]
        
            for c in range(len(components)):
                ### take Residual of the component
                rs=dfS[components[c]]-dfS[Reference+'_'+components[c]] #'GrAtSiD_'
                matrix_median[st,indici2,c]=rs[indici1]
            
            if (st/len(stations_to_use))*100 > ten_step:
                print(str(ten_step)+'%')
                ten_step+=10
        
        print("Compute median")
        median_resT=[]
        median_resWT=[]
        for c in range(len(components)):              
            ##### the wheights are based on the temporal length of 1 time series and its distance
            combined_weights=len_athN*distance_athN
            median_res=np.zeros([len(t)])
            median_resW=np.zeros([len(t)])

            ###### compute the median
            for kk in range(matrix_median.shape[1]):
                indici_zero=np.where(matrix_median[:,kk,c]!=0)[0]
                median_res[kk]=np.nanmedian(matrix_median[indici_zero,kk,c])  
                if len(indici_zero)==0:  
                    median_resW[kk]=np.nan
                else:
                    #print(weighted_median(matrix_median[indici_zero,kk], combined_weights[indici_zero]))
                    median_resW[kk]=weighted_median(matrix_median[indici_zero,kk,c], combined_weights[indici_zero])
    
            ###### if there are some nans, then fill with 0
            if np.isnan(median_res).any():
                print(max_consecutive_nan(median_res))
                indexes=[ind[0] for ind in np.argwhere(~np.isnan(median_res))] 
                median_res[indexes]=0
                median_resW[indexes]=0
                
                #f = interpolate.interp1d(t[indexes], median_res[indexes],fill_value="extrapolate")
                #median_res = f(t)
                #fW = interpolate.interp1d(t[indexes], median_resW[indexes],fill_value="extrapolate")
                #median_resW = fW(t)

            median_resWT.append(median_resW)
            median_resT.append(median_res)
    
        median_resWT=np.array(median_resWT)  
        median_resT=np.array(median_resT)  

        print(data.shape,median_resT.shape,median_resWT.shape)
        filtered=np.vstack([np.transpose(data),median_resT,median_resWT])
        filtered=np.transpose(filtered)

        namesTM=[]
        namesTW=[]
        for c in components:
            namesTM.append(['Med_'+c])
            namesTW.append(['MedW_'+c])
                
        namesTM = [item for sublist in namesTM for item in sublist]
        namesTW = [item for sublist in namesTW for item in sublist]
            
        new_cols=new_cols+namesTM+namesTW
        dfCMF = pd.DataFrame(filtered[:] , columns=new_cols[:])
        dfCMF['YYMMDD']= pd.to_datetime(dfCMF['YYMMDD']).astype('datetime64[ns]')
        datetime_index = pd.DatetimeIndex(dfCMF.YYMMDD)
        
        # Check for duplicates
        assert not datetime_index.duplicated().any(), "Datetime series contains duplicates."
        # Check if all dates are increasing
        assert (datetime_index == datetime_index.sort_values()).all(), "Dates in the datetime series are not in increasing order."

        if save_flag==True:  
            #date columns as first
            dfCMF.to_csv(save_folder+'/'+str(station)+'.txt', header=None, index=None, sep=' ', mode='a')
    else:
        print ('CMF cannot be applyed - the station is isolated')
    
    print("Finished")
    return dfCMF

def distance_loop(df,save_flag=False,cd=None):
    
    """
    Parameters
    ----------
        df: file of coordinates with columns=['station','latitude','longitude','altitude']
        save_flag: bolean (if True save the list)
        
     Returns 
    ----------
        Distance_list: list of lists, where each list includes all stations at a distance shorter than a thresholds
    """
    
    Distance_list=[]
    for i in range(len(df)):
        df_new=df[df.station!=df.station.iloc[i]]  
        distanceT=[]
        for st in range(len(df_new)): 
            dis=distance([df_new.latitude.iloc[st],df_new.longitude.iloc[st]], [df.latitude.iloc[i],df.longitude.iloc[i]])
            if dis< thr:
                distanceT.append([df_new.station.iloc[st],dis])
        Distance_list.append(distanceT)
   
    if save_flag==True:
        with open(cd+'Distance_list', "wb") as fp:   #Pickling
            pickle.dump(Distance_list, fp)
            
    return Distance_list

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
    
def build_correlation_matrix(df,soln_folder_path,new_cols,Reference,save_flag=False,save_folder=True):
    
    """
    Parameters
    ----------
        df: file of coordinates with columns=['station','latitude','longitude','altitude']
        cd: folder where you have the txt file
        new_cols: names of the columns of a txt file
        Reference: string - name of the trajectory you would like to use ('E.g. GrAtSiD or DL')
        save_flag: bolean (if True save the numpy array)
        save_folder: folder where save the correlation matrix

    Returns
    ----------
        corr: Correlation matrix with a dimension of [len(stations),len(stations)-1,len(componets)]
              the last three values are: distance between stations [0], number days in common [1], Pearson correlation coefficient for the n components [2,3,4]
    """
    
    if not any('GrAtSiD' in element for element in new_cols) or not any('DL' in element for element in new_cols):
        raise ValueError("The string 'GrAtSiD' or 'DL' is not present in the input columns")
    if not any('E' or 'U' or 'N' in element for element in new_cols):
        raise ValueError("At least one component has to be present in the input columns")
    if not any('station' in element for element in list(df.columns)):
        raise ValueError("The string 'station' is not present in the input columns in the file of the stations coordinates ")

    #### Take coordinates of stations that are inside the folder of interest
    stations=id_names_txt(soln_folder_path)
    df = df[df['station'].isin(stations)]
   
    ten_step=5
    n=len(df)
    corr=np.zeros([len(df),len(df)-1,5]) 

    for i in range(len(df)):
        ### Load the dataframe of the reference station
        station=df.station[i]
        file_I=soln_folder_path+'/'+str(station)+'.txt'
        dfs = pd.read_csv(file_I,delim_whitespace=True,header=0,on_bad_lines='skip', skiprows=check_rows(file_I,new_cols))
        new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
        dfs.rename(new_names_map, axis=1, inplace=True)
        t=list(dfs.YYMMDD)
        
        df_new=df[df.station!=station]  
        for st in range(len(df_new)): 
            file_I=soln_folder_path+'/'+str(df_new.station.iloc[st])+'.txt'
            dfS = pd.read_csv(file_I, delim_whitespace=True,header=0,on_bad_lines='skip',skiprows=check_rows(file_I,new_cols))
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
                    r=dfs[components[c]]-dfs[Reference+'_'+components[c]]
                    rs=dfS[components[c]]-dfS[Reference+'_'+components[c]]
                    cc=np.corrcoef(rs[indici1],r[indici2])
                    corr[i,st,2+c]=cc[0,1]
        
        if (i/n)*100 > ten_step:
            print(str(ten_step)+'%')
            ten_step+=5
    if save_flag==True:
        np.save(save_folder+'Correlation_matrix',corr)
        
    return corr

def denoise(comp,t,d,r,soln_folder_path,station,thr_distance,cd_base,cd_data,new_cols,weight_flag=None):

    """
    Return the CMC residuals
    
    Parameters
    ----------
       comp: component (e.g. 'E')
       t: time vector
       d: raw data
       r: original residual
       soln_folder_path: folder of all stations employed
       Reference: string - name of the trajectory you would like to use ('E.g. GrAtSiD or DL')
       station: name of the station to denoise
       thr_distance: distance threshold
       cd_base: folder of the correlation matrix
       cd_data: folder of the data
       new_cols: names of the columns of a txt file
       weight_flag: if True the median is weighted 

    Returns
    ----------
       t,r,d,median_res/median_resW
    """
    if not any('GrAtSiD' in element for element in new_cols) or not any('DL' in element for element in new_cols):
        raise ValueError("The string 'GrAtSiD' or 'DL' is not present in the input columns")
         
    if comp!='E' or  comp!='N' or  comp!='U':
        raise ValueError("component must be a string like 'E' or 'N' or 'U' ")
    
    corr=np.load(cd_base+'Corr_'+comp+'.npy')
    if comp=='E':
        ind=2
    elif comp=='E':
        ind=3
    elif comp=='U':
        ind=4
        
    corr=corr[:,:,ind]

    stations=id_names_txt(soln_folder_path)
    df = df[df['station'].isin(stations)]
    
    # Index of the station
    ii=np.argwhere(np.array(stations)==station)[0][0]
    ii=df.index[df.station == station].tolist()[0]
    stations_new = stations_new[:ii] + my_list[ii + 1:]

    ##### Correlation has to be positive #####
    indiciS=np.where((corr[ii,:,1]>0.1))[0]
    corre_ath=corr[ii,indiciS,1] #correlation
    len_ath=corr[ii,indiciS,2] #intersect_length
    distance_ath=corr[ii,indiciS,0] #distance
    stations_new=stations_new[indiciS]

    ##### Distance has to be positive #####
    above_th=np.where(distance_ath<thr_distance)[0]
    corre_ath=corre_ath[above_th]
    len_ath=len_ath[above_th]
    distance_ath=distance_ath[above_th]
    stations_new=stations_new[above_th]  
    
    ##### if no it means CMF cannot be applyed - the station is isolated
    if len(len_ath)!=0:
        ##### I normalize intersect_length - distance
        len_athN=(len_ath - min(len_ath)) / (max(len_ath) - min(len_ath))
        distance_athN=(distance_ath - min(distance_ath)) / (max(distance_ath) - min(distance_ath))

        ##### Here I allocate the residual of the other stations
        matrix_median=np.zeros([len(stations_new),len(t)])
        n=len(stations_new)
        ten_step=10

        ##### I Start the loop 
        for st in range(len(stations_new)): 
            file_I=cd_data+comp+'/'+stations_new[st]+'.txt'
            dfS = pd.read_csv(file_I,delim_whitespace=True,header=0,on_bad_lines='skip', skiprows=check_rows(file_I,new_cols))
            new_names_map = {dfS.columns[i]:new_cols[i] for i in range(len(new_cols))}
            dfS.rename(new_names_map, axis=1, inplace=True)  
            ts=list(dfS.YYMMDD)
            
            indici1=np.nonzero(np.in1d(ts, t))[0]
            if len(indici1)>0:
                indici2=np.nonzero(np.in1d(t, ts))[0]
                rs=dfS[comp]-dfS[Reference+'_'+comp]
                matrix_median[st,indici2]=rs[indici1]
   
            if (st/n)*100 > ten_step:
                print(str(ten_step)+'%')
                ten_step+=10
                
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
            median_res[indexes]=0
            median_resW[indexes]=0
            #f = interpolate.interp1d(t[indexes], median_res[indexes],fill_value="extrapolate")
            #median_res = f(t)
            #fW = interpolate.interp1d(t[indexes], median_resW[indexes],fill_value="extrapolate")
            #median_resW = fW(t)
        
        print("Finished")
        if weight_flag==True:   
            return t,r,d,median_resW
        else:
            return t,r,d,median_res
    else:
        print ('CMF cannot be applyed - the station is isolated')
        return t,r,d,0

def plot_CMF_correlation(t,r,d,median_res,comp,station):

    """
    Make a plot of CMF results

    Parameters
    ----------
        t: vetorized time
        r: real residuals 
        d: raw time-series  
        median_res:
        comp: component 
        station: name of the station
    
    """
    
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

def plot_CMC_dataframe(dfs,station,Reference,comp,weight_flag=False,save_flag=False,cd_save=None):

    """
    Make a plot of CMF results using a dataframe as input (the output of the CMF function)

    Parameters
    ----------
        dfs: dataframe of denoised results
        station: name of the station
        comp: components to use (eg: 'E')
        Reference: string - name of the trajectory you would like to use ('E.g. GrAtSiD or DL')
        weight_flag: if True returns the weighted median else the basic median
        save_flag: bolean if True save the figure
        cd_save: folder where save the figure
   
    Returns
    ----------
        The plot of CMF results
    """
    
    r=np.array(dfCMF[comp]-dfCMF[Reference+'_'+comp]).astype('float')
    d=np.array(dfCMF[comp]).astype('float')
    if weight_flag==True:
        median_res=np.array(dfCMF['MedW_'+comp])
    else:
        median_res=np.array(dfCMF['Med_'+comp])
       
    datetime_list = dfs.YYMMDD
    t = np.array([(dt - datetime_list.iloc[0]).days for dt in datetime_list]).astype('int')
    ### Remove the trend
    from scipy.stats import linregress
    trend=linregress(t,d)
    trend_vector=t*trend.slope+trend.intercept

    fig,axes=plt.subplots(3,2,figsize=(10,8))
    fig.subplots_adjust(wspace=.2,hspace=0.32)
    c=0

    axes[0,0].scatter(datetime_list,r,s=10,facecolor='c',linewidth=0.1,edgecolors='k')
    axes[0,0].set_title('Original Residuals Time Series  - '+comp+' - '+station)
    axes[0,1].set_title('Original Time series - '+comp+' - '+station)
    axes[0,1].scatter(datetime_list,d-trend_vector,facecolor='c',s=10,linewidth=0.1,edgecolors='k')

    axes[1,0].scatter(datetime_list,r,s=10,facecolor='c',linewidth=0.1,edgecolors='k')
    axes[1,0].set_title('Comparison  - '+comp+' - '+station)
    axes[1,0].scatter(datetime_list,r-median_res,s=5,facecolor='salmon',linewidth=0.1,edgecolors='k')
    axes[0,0].text(.5,.04,r'$\mu$ ='+str(round(np.mean(r),8))+' $\pm$ '+str(round(np.std(r),4))+'$\sigma$',horizontalalignment='center',transform= axes[0,0].transAxes,fontsize=8,
    bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'),color='b')

    axes[2,0].set_title('Denoised Residuals Time series - '+comp+' - '+station,color='salmon')
    axes[2,0].scatter(datetime_list,r-median_res,s=10,facecolor='salmon',linewidth=0.1,edgecolors='k')
    axes[2,0].text(.5,.04,r'$\mu$ ='+str(round(np.mean(r-median_res),8))+' $\pm$ '+str(round(np.std(r-median_res),4))+'$\sigma$',horizontalalignment='center',transform= axes[2,0].transAxes,fontsize=8,
    bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'),color='r')

    axes[1,1].set_title('Comparison - '+comp+' - '+station)
    axes[1,1].scatter(datetime_list,d-trend_vector,facecolor='c',s=10,linewidth=0.1,edgecolors='k')
    axes[1,1].scatter(datetime_list,d-median_res-trend_vector,s=5,facecolor='salmon',linewidth=0.1,edgecolors='k')

    axes[2,1].set_title('Denoised Time series - '+comp+' - '+station,color='salmon')
    axes[2,1].scatter(datetime_list,d-median_res-trend_vector,s=10,facecolor='salmon',linewidth=0.1,edgecolors='k')

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
            axes[k,j].set_xlim([datetime_list.iloc[0],datetime_list.iloc[-1]])
            if k<2:
                axes[k,j].set_xticklabels([])

    if save_flag==True:
        fig.savefig(cd_save+'CMC_'+str(comp)+'_'+station+'.pdf')    
        fig.savefig(cd_save+'CMC_'+str(comp)+'_'+station+'.png') 
    
    return plt.show()

def plot_correlation_matrix(corr,save_flag=False,save_folder=None):
    
    """
    Parameters
    ----------
       corr: Correlation matrix with a dimension of [len(stations),len(stations)-1,len(componets)]
             the last three values are: distance between stations [0], number days in common [1], Pearson correlation coefficient for the three components [2,3,4]
       save_flag: bolean (if True, save the plot)
       save_folder: folder where save the figure
       
    Returns
    ----------
       Scatter plot for each components the Pearson correlation coefficient as a function of the Intersation Distance 
             the scatter color depicts the number of days in common between each time-series 
    """
    
    from scipy.interpolate import splev, splrep

    fig,axes=plt.subplots(corr.shape[2]-2,1,figsize=(9,10))
    fig.subplots_adjust(wspace=.32,hspace=0)
    c=0
    
    for ax in axes.flat:
        #find time series with temporal intersections
        indici=np.where(corr[c,:,1]!=0)[0]
        distance = np.hstack( np.array(corr,dtype='object')[:,indici,0]).astype('float')
        intersect_len = np.hstack( np.array(corr,dtype='object')[:,indici,1]).astype('int')
        correlation = np.hstack( np.array(corr,dtype='object')[:,indici,2+c]).astype('float')
        
        #Remove Nans
        no_nan=np.where(~np.isnan(correlation))
        distance=distance[no_nan]
        correlation=correlation[no_nan]
        intersect_len=intersect_len[no_nan]

        # I normalize the lenintersect
        intersect_len_N=(intersect_len - min(intersect_len)) / (max(intersect_len) - min(intersect_len))

        # I revert the lenintersect [shorter series higher uncertanty]
        ww=intersect_len_N.copy()
        indici_sort=sorted(range(len(intersect_len_N)), key=lambda k: intersect_len_N[k])
        intersect_len_N[indici_sort]=sorted(ww, reverse=True)
    
        #sort distance to make the fit
        dis_sort=np.argsort(distance)
        sorted_dis=[distance[i] for i in dis_sort]
        corr_sort=[correlation[i] for i in dis_sort]
        ww=[intersect_len_N[i] for i in dis_sort]
    
        #distances have to be unique
        indixes_unique_dis=np.unique(sorted_dis, return_index=True)
        ww=[ww[i] for i in indixes_unique_dis[1]]
    
        #make the fit
        spl = splrep(indixes_unique_dis[0], [corr_sort[i] for i in indixes_unique_dis[1]],w=ww,k=4) #,w=intersect_len_N
        x2 = np.linspace(0, max(distance), 10000)
        y2 = splev(x2, spl)
    
        #plot the scatter
        scatt=ax.scatter(distance,correlation,s=0.1,c=intersect_len) 
        ax.plot(x2, y2, color='r',
                label='fit')
        ax.set_ylabel('Pearson Correlation')
        ax.set_xlabel('Distance [km]')
        #ax.set_ylim([-1,1])
        ax.set_xlim([0,max(distance)])
        if c!=len(components)-1:
            axes[c].set_xticks([])

        ax.text(.5,.04,'Component: '+components[c],horizontalalignment='center',transform= ax.transAxes,fontsize=7,
        bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'),color='k')
        print(c)
        c+=1

    cbar=fig.colorbar(scatt, ax=axes.ravel().tolist())
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_title('Intersection [days]', rotation=0, fontdict = {"size":8})

    if save_flag==True:
        fig.savefig(save_folder+'/Correlation_Residuals.pdf')    
        fig.savefig(save_folder+'/Correlation_Residuals.png',dpi=300) 
    
    return plt.show()

""" 
################### Example if you don't have the Correlation Matrix #######################
#coordinates file
import pandas as pd
import numpy as np
import datetime,math

##### Coordinates file #####
dfC='/Users/giacomo/Documents/PhD/Papers/GNSS_DENOISER/New_zeland_C/stations_coordinates.txt'
dfC = pd.read_csv(dfC, delimiter=',',names=['station','latitude','longitude','altitude'],header=None)

##### Reference station #####
ref_station='AHTI.txt'
## Folder of stations to use ##
soln_folder_path='/Users/giacomo/Documents/PhD/Papers/GNSS_DENOISER/New_zeland_C/Filtered_31_28'
file=soln_folder_path+ref_station

##### Components #####
components=['E','N']

gratsid_flag=True 
exp_flag=True
namesT=[]
if gratsid_flag==True and exp_flag==True: 
    for c in components:
        namesT.append([c,'DL_'+c,'EMV_'+c,'GrAtSiD_'+c])
else:
    for c in components:
        namesT.append([c,'DL_'+c])
new_cols = [item for sublist in namesT for item in sublist]
new_cols.insert(0, 'YYMMDD') 

### You wanto to base your CMF on the DL model or on 'GrAtSiD ###
Reference='DL'
##### Run CMF #####
thr_distance=2000 #distance threeshold
dfCMF=CMF(file,dfC,soln_folder_path,3000,new_colsReference)                              

##### Build and Plot the correlation Matrix #####
dfC = pd.read_csv(dfC, delimiter=',',names=['station','latitude','longitude','altitude'],header=None)
save_folder='/Users/giacomo/Documents/PhD/Papers/GNSS_DENOISER/New_zeland_C/'
corr=build_correlation_matrix(dfC,soln_folder_path,new_cols,Reference,components,save_flag=True,save_folder=save_folder)
corr=np.load(save_folder+'Correlation_matrix.npy')
plot_correlation_matrix(corr)

################### Example if you have the Correlation Matrix #######################
comp='U'
cd_base='/home/giacomo/Documents/Denoiser_GPS/Common_mode_analysis/'
cd_data='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/t_disps_resids/'
#import list of all stations
dfC = pd.read_csv('/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/Stations_coordinates.txt', delimiter=',',names=['station','latitude','longitude','altitude'],header=None)
station='CABU'
thr_distance=3000
save_flag=False
cd_save='/home/giacomo/Documents/Denoiser_GPS/Common_mode_analysis/Figures/pdf/'

### You wanto to base your CMF on the DL model or on 'GrAtSiD ###
Reference='GrAtSiD'

##### Grab time series of the station to denoise
t=np.loadtxt(cd_data+comp+'/'+station+'.txt')[:,0]
d=np.loadtxt(cd_data+comp+'/'+station+'.txt')[:,1]
r=np.loadtxt(cd_data+comp+'/'+station+'.txt')[:,2]

t,r,d,median_res=denoise(comp,t,d,r,df,Reference,station,thr_distance,cd_base,cd_data,new_cols)
plot_CMC_correlation(t,r,d,median_res,comp,station)
"""
