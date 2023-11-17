#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:45:21 2022

some functions to be called in this workign folder of deep learning of residuals


@author: jon
@author: giacomo
"""

import sys ,glob, os,pickle, warnings,random
sys.path.append('/home/giacomo/Documents/Denoiser_GPS/sharing_gratsid_tf_in_development')
sys.path.append('/home/giacomo/Documents/Step_model')
sys.path.append('/home/giacomo/Documents/Denoiser_GPS/Denoiser_code')
from funcs_4_DL_resids import *
from gratsid_tf_gpu_functions_SHARED import *
from Build_step_indexes import find_step

import numpy as np
import scipy.interpolate as interpolate
import datetime as dt


def build_txt(data_folder_path,soln_folder_path,save_path,components):
    
    """
    This function create txt files of raw data and residuals from gratsid table solutions
    
    Parameters
    ----------
       data_folder_path: folder of txt input files
       soln_folder_path: folder of gratsid tables
       save_path: folder where saving
       components: components to use
       
       example:
       data_folder_path = '/Users/giacomo/Documents/PhD/SL_SL_Forecast/Cascadia/Stations_GRATSID/'
       soln_folder_path = cd+'/sols_tables_'
       save_path= cd+'/t_disps_resids'
       components=['E/','N/','U/']
    
    Returns
    ----------
       save txt files for each station. Each file has 3 columns: vectorized time - raw data - residuals
    
    """
    ############ Recreate prexisting save_path directory ############
    isExist = os.path.exists(save_path)
    if not isExist:
        os.mkdir(save_path)
    
    #### Remove stations that  not have all the components ####
    ### Consider before npz files - gratsid tables ####
    # remove single component npz files    
    for c in range(len(components)-1):   
        quali=id_names_npz(soln_folder_path+components[c])
        quali1=id_names_npz(soln_folder_path+components[c+1])
        diff1=list(set(quali) - set(quali1))
        diff2=list(set(quali1) - set(quali))
        for j,k in zip(diff1,diff2):
            os.remove(soln_folder_path+components[c]+str(j)+'.npz')
            os.remove(soln_folder_path+components[c+1]+str(k)+'.npz')

    quali=id_names_npz(soln_folder_path+components[0])
    quali1=id_names_npz(soln_folder_path+components[2])
    diff1=list(set(quali) - set(quali1))
    diff2=list(set(quali1) - set(quali))
    for j,k in zip(diff1,diff2):
        os.remove(soln_folder_path+components[0]+str(j)+'.npz')
        os.remove(soln_folder_path+components[2]+str(k)+'.npz')

    ############ pick list of txt file names ############
    names_txt=id_names_txt(data_folder_path) #from data_bank
    
    #npz files list for all components - equal for all components
    names=id_names_npz(soln_folder_path+components[0])
    
    # remove txt files or npz files in excess
    if len(names)!=len(names_txt):
        if len(names)>len(names_txt):
            da_eliminare=list(set(names) - set(names_txt))
            print('Data bank folder and npz folder have a different number of files: the npz files in in excess are: '+ str(da_eliminare)) 
            flag=input('Should I remove them? Y or N?:')
            if flag=='Y':
                for sta in da_eliminare:
                    for c in range(len(components)):  
                        os.remove(soln_folder_path+components[c]+str(sta)+'.npz')
        if len(names_txt)>len(names):
            da_eliminare=list(set(names_txt) - set(names))
            print('Data bank folder and npz folder have a different number of files: the txt files in in excess are: '+ str(da_eliminare))
            flag=input('Should I remove them? Y or N?:')
            if flag=='Y':
                for sta in da_eliminare:
                    for c in range(len(components)):  
                        os.remove(data_folder_path+str(sta)+'.txt')

    # Assert
    for c in range(len(components)-1):   
        assert len(id_names_npz(soln_folder_path+components[c])) == len(id_names_npz(soln_folder_path+components[c+1])) == len(id_names_txt(data_folder_path))

    ########## Loading the data and solutions and saving what we need (t,disps,resids) ##################

    for c in range(len(components)):

        ##### step list #####
        step_listC=[]

        ############ finally build ############
        soln_folder=soln_folder_path+components[c]
        save_pathc=save_path+'/'+components[c]
        sol_path = soln_folder+names[0]+'.npz'
        options = np.load(sol_path,allow_pickle=True)['options']
        options = options.item()
        data_cols = np.load(sol_path)['data_cols']
        if os.path.isdir(save_path+'/'+components[c]):
            for f in os.listdir(save_path+'/'+components[c]):
                os.remove(os.path.join(save_path+'/'+components[c], f))
        else:
            os.mkdir(save_path+'/'+components[c])
         
        ########## Allocate  ##########
        for i in range(len(names)):
            sol_path = soln_folder+names[i]+'.npz'
            data_path = data_folder_path+names[i]+'.txt'
            out_path = save_pathc+names[i]+'.txt'
    
            ## loading in the data and the corresponding gratsid solution table
            data = np.loadtxt(data_path)
            perm = list(np.load(sol_path,allow_pickle=True)['perm'])
            sols = list(np.load(sol_path,allow_pickle=True)['sols'])
    
            ## converting time to python datetime integer and isolating the fit directional components
            t = gen_jjj(data[:,0].astype(int),data[:,1].astype(int),data[:,2].astype(int))
            y = data[:,data_cols] ## columns 3,4,5  (in python indexing) are E,N,U
    
            ## getting the fits
            signal = fit_decompose(t,y,None,options['tik_mul'], \
                sols,np.asarray(perm),options['bigTs'],options['Fs']) 
            
            #### residuals
            resid = np.nanmedian(np.array(signal[-1]),axis=0)
            
            ####################### THRESHOLD FOR STEP #######################
            threshold=resid.std()*2
            ######## getting steps ########
            step_indixes,velocity_step=find_step(signal,t,thr=threshold) 
            if len(step_indixes)>0:
                step_indixes=t[step_indixes]
            
            ## allocate into the list  ##
            step_listC.append([names[i],step_indixes])
            
            #### Saving
            out = np.hstack([t[:,None],y,resid,velocity_step[:,None]]) 
            ######  Save raw data and residuals ######
            np.savetxt(fname=out_path,X=out)
        
        ## save the list ##
        with open(save_path+components[c].split('/')[0]+'_list_steps.pkl', 'wb') as f:
                pickle.dump(step_listC, f)
            
    return


def id_names_npz(soln_folder_path):
    """
    Return list of the names of the *npz files inside a folder
    
    Parameters
    ----------
       soln_folder_path: input folder 
   
    Returns
    ---------- 
       List of the names of the files
    """
    
    os.chdir(soln_folder_path)
    names=[]
    for file in glob.glob("*.npz"):
        names.append(file.split('.')[0])
    return sorted(names) 


def id_names_txt(soln_folder_path):
    """
    Return list of the names of the *txt files inside a folder
    
    Parameters
    ----------
       soln_folder_path: input folder 
    
    Returns
    ---------- 
        List of the names of the files
    """
    
    os.chdir(soln_folder_path)
    names=[]
    for file in glob.glob("*.txt"):
        names.append(file.split('.')[0])
    return sorted(names) 


def elongate_and_interpolate(t_in,d_in,r_in,h_in,max_gap):
    
    """
    This function Interpolate time series
    
    Parameters
    ----------
       t_in: time vector
       d_in: raw data vector
       r_in: residuals vector
       h_in: heaviside vecotr
       max_gap: maximum gap for having nans in the series (int)
    
    Returns
    ----------
       t: vectorized time vector
       d: interpolated raw data vector
       r: interpolated residuals vector
       h: interpolated heaviside vector
    
    """
    
    noise_std = np.nanstd(r_in)
    
    ####################################################################################
    ## First elongating
    t = np.arange(t_in[0],t_in[-1]+1,1).astype(int)
    d = np.zeros(t.shape)
    d.fill(np.nan)
    r = np.zeros(t.shape)
    r.fill(np.nan)
    h = np.zeros(t.shape)
    h.fill(np.nan)
    
    ii = t_in.astype(int)-t[0]
    d[ii] = d_in
    r[ii] = r_in
    h[ii]= h_in
    
    ####################################################################################
    ## Getting indices of the starting and ending of nan-gaps
    nan_bool = 1*np.isnan(d)
    
    starts = np.arange(t.size)[1:][nan_bool[1:]-nan_bool[0:-1] == 1]
    ends = np.arange(t.size)[0:-1][nan_bool[1:]-nan_bool[0:-1] == -1]
    
    ## Getting rid of cases where nan-gaps have starts but no ends
    keep_starts = starts<=ends.max()
    starts = starts[keep_starts]
    
    
    ## Getting rid of cases where nan-gaps have ends but no starts
    keep_ends = ends>=starts.min()
    ends = ends[keep_ends]
    
    s_e = np.hstack([starts[:,None],ends[:,None]])

    ####################################################################################
    # Counting the gaps
    
    count = np.cumsum(nan_bool)
    gap = count[ends]-count[starts]+1
    s_e_gap = np.hstack([s_e,gap[:,None]]) 
    

    ####################################################################################
    # Interpolating
    
    ## Defining which data the interpolatin are modelled on
    fd = interpolate.interp1d(t[~np.isnan(d)], (d-r)[~np.isnan(d)],kind='linear')
    #fd = interpolate.interp1d(t[~np.isnan(d)], d[~np.isnan(d)],kind='linear')
    
    
    ## defining the time vector where new points are calculated
    ii = np.array([])
    for i in range(s_e.shape[0]):
        if s_e_gap[i,-1] <= max_gap:
            ii = np.append(ii,np.arange(s_e[i,0],s_e[i,1]+1))
    
    ii = ii.astype(int)
    
    new_m = fd(t[ii]) 
   
    r[ii] = noise_std*np.random.randn(ii.size) ### Putting the interpolated values into y
    d[ii] = new_m+r[ii] ### Putting the interpolated values into y
    h[ii]=0

    return t,d,r,h


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
    
    ##### Create the 1-0 vector #####
    H_vector_zo=np.zeros(H_vector.shape)
    
    threshold=np.nanstd(Y)*2
    
    ############ 2 std(resid) is the threshold velocity  ############
    ind=np.argwhere(H_vector>threshold)
    ## put 1 where there is a step ##
    H_vector_zo[ind[:,0],ind[:,1]]=1
    
    ##### removing the median #####
    X = X - np.repeat(np.nanmedian(X,axis=1)[:,None],X.shape[1],axis=1)  
    
    return X,Y,H_vector_zo,H_vector
    
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
            
            ### interpolating only if there are missing gaps (nans values inside)
            if np.isnan(d_in).any() or len(np.argwhere(np.diff(t_in)>1)):
                t,d,r,h = elongate_and_interpolate(t_in,d_in,r_in,h_in,max_gap)
            else:
                t,d,r,h=t_in,d_in,r_in,h_in
        
            #### Making X and y
            X,Y,H_vector,H_Vel = XY_from_d_and_length(d,r,h,input_length)

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
            np.savez(file=save_path_s,X=X,Y=Y,H=H_vector,t=t,d=d,r=r,h=h,H_Vel=H_Vel,list_transient=list_transient)
    return    
       
def create_XY_for_generator(load_path,save_path,components,perctr,percval,directories,write_txt=True):
    
    """
    Create .npy files from .npz file and shuffle txt files which are needed by the generator
    The data are splitted based on number of stations
    
    Parameters
    ----------
       load_path: folder of .npz files ( '/Users/giacomo/Documents/PhD/DENOISER_GPS/'Cascadia/tXY_by_component')
       save_path: folder where saving (generator) ( '/Users/giacomo/Documents/PhD/DENOISER_GPS/'Cascadia')
       components: components to use
       perctr: percentage of training
       percval: percentage of validation
       directories: ['/TRAINING','/VALIDATION','/TESTING']
       write_txt: if True write txt files needed by generator (bolean) 
    
    Returns
    ----------
        Save X.npy and Y.npy files for each component of each station
           X.npy file: input (raw data)
           Y.npy file: Target (residuals) 

        Creates a Indexes.txt file for TRAINING, VALIDATION, TESTING'
        Each line of a txt file is structured as follows:
           Network    Component    Station    Sample
           CASCADIA   E            P426       4460

        lengths: List of lists where each element of List is for one station
             for example: List[0] includes two elements: [Name of the station, number of examples for station]

    """
    #### area #####
    area=save_path.split('/')[-1]
    from random import shuffle
    
    #### Shuffle order of the stations ####
    names=id_names_npz(load_path+'/'+components[0])
    shuffle(names)
    
    #### Training-validation-testing split ####
    start=0
    startval=int((len(names)/100)*perctr) 
    endval=int((len(names)/100)*percval)+startval 
    starters=[start,startval,endval,len(names)-1]
    
    lengths=[[] for i in range(len(directories))]

    isExist = os.path.exists(save_path+'/Generators')
    if not isExist:
        os.mkdir(save_path+'/Generators')
    
    #### Loop into the dataset ####
    for d in range(len(directories)):
        print(directories[d].split('/')[1])
        names_new=names[starters[d]:starters[d+1]]

        #### Loop into the components ####
        for c in range(len(components)):
            print(components[c])
            
            #### Update the paths and check if they exist
            load_path_component=load_path+'/'+components[c]
            save_path_component=save_path+'/Generators/'+components[c]
            isExist = os.path.exists(save_path_component)
            if not isExist:
                os.mkdir(save_path_component)
            
            ten_step=10
            #### Loop into stations #### 
            for i in range(len(names_new)):
                perc=i*100/len(names_new)
                if perc>ten_step:
                    print(str(round(perc))+'%')
                    ten_step+=10

                save_path_stat=save_path_component+names_new[i]
                
                #### Remove prexisting files if they exist
                # SAVE DECOMENNTA
                
                if os.path.exists(save_path_stat):
                     for f in os.listdir(save_path_stat):
                        os.remove(os.path.join(save_path_stat, f))
                else:
                    os.mkdir(save_path_stat)

                #### Load data #### 
                in_path = load_path_component+names_new[i]+'.npz'
                t = np.load(in_path)['t']
                X = np.load(in_path)['X']
                Y = np.load(in_path)['Y']
                list_transient=np.load(in_path)['list_transient']

                if X.size==0 or Y.size==0:
                     warnings.warn('Warning '+load_path_component+names_new[i]+'.npz'+' is empty!!!')
                     os.remove(os.path.join(load_path_component, names_new[i]+'.npz')) ###delete the file
                else:

                    #### no nan indexes
                    nonan_ind=[ind[0] for ind in np.argwhere(~np.isnan(X).any(axis=1))]
                    nonan_indy=[ind[0] for ind in np.argwhere(~np.isnan(Y).any(axis=1))]
                    
                    assert nonan_ind==nonan_indy ### just to confirm
                    
                    nonanX=X[nonan_ind]
                    nonanY=Y[nonan_indy]

                    #### no nan time
                    t_nonan=t[nonan_ind]
                    
                    #### Create npy file only if non nans are present #### 
                    for ii in range(nonanX.shape[0]):
                        vett_time=list(range(int(t_nonan[ii]-nonanX.shape[1]),int(t_nonan[ii])))

                        #### this example contains a transient ####
                        ### append_list is a string where each element separated by a - represents the position of the transient in the window
                        if any(i in vett_time for i in list_transient):
                            inter=np.intersect1d(vett_time,list_transient)
                            append_list=str(np.where(np.array(vett_time)==np.array(inter[0]))[0][0]) #postion transient in the window
                            for ij in range(1,len(inter)):
                                append_list=append_list+'-'+str(np.where(np.array(vett_time)==np.array(inter[ij]))[0][0])
    
                            with open(save_path+'/Indexes_functions'+directories[d].split('/')[1]+'.txt', "a") as file:
                                file.write(area+','+components[c].split('/')[0]+','+names_new[i]+','+str(ii)+',T,'+str(append_list)+'\n') 
            
                        #### this example does not contain a transient ####
                        else:
                            with open(save_path+'/Indexes_functions'+directories[d].split('/')[1]+'.txt', "a") as file:  
                                file.write(area+','+components[c].split('/')[0]+','+names_new[i]+','+str(ii)+',N,'+str(00)+'\n')  
                        
                        ################################################################################################################################################################# 
                        # SAVE DECOMENNTA
                        np.savez(save_path_stat+'/'+str(ii), X=nonanX[ii,:], Y=nonanY[ii,:])
                        
                        #### Create txt file #### 
                        with open(save_path+'/Indexes'+directories[d].split('/')[1]+'.txt', "a") as file:  
                           file.write(area+','+components[c].split('/')[0]+','+names_new[i]+','+str(ii)+ '\n')   

                    if c==0:
                        lengths[d].append([names_new[i],nonanX.shape[0]])
        
        #### Shuffle indexes of txt files ####
        path_indexes=save_path+'/Indexes'+directories[d].split('/')[1]+'.txt'
        shuffle_txt(path_indexes)
        shuffle_txt(save_path+'/Indexes_functions'+directories[d].split('/')[1]+'.txt')
         
    DATASETS=[save_path.split('/')[-1]]
    DATA=Data_to_use(save_path,directories,DATASETS,components,nonanX.shape[0])
    ############### Create list necessary by the generator #################
    list_filesTot=DATA.create_list_files(save_path)
    ##### Save tables as pickle #####
    with open(save_path+'/indixes_list_generator.pkl', 'wb') as f:
        pickle.dump(list_filesTot, f)
            
    return lengths
     
def shuffle_txt(path_indexes):
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
    indices=list(range(len(lines)))
    shuffle(indices)
    lines=lines[indices]
    open(path_indexes, 'w').writelines(lines)

    return indices 

def exists(var):
    """
    Check if a variable exist

    Parameters
    ----------
       var: variable to check
    """
    return var in globals()


def Scaler(cd_base,list_files,Batch_size,input_length):

    """
    Scaling the dataset 
    
    Parameters
    ----------
       cd_base: bank directory
       list_files: list of training files
       Batch_size: batch size
                   choose it depending on your memory capability
       input_length: length of one target_file 
    
    Returns
    ----------
       The Scaler object
    
    """

    from sklearn.preprocessing import StandardScaler
    sca = StandardScaler()
    
    scaler = StandardScaler()
    n = len(list_files)
    ten_step=10
    
    index = 0  # helper-var
    
    while index < n:
        partial_size = min(Batch_size, n - index)  # needed because last loop is possibly incomplete
        X = np.empty((Batch_size, input_length))
        j=0
        for i in range(index,Batch_size+index):
            X[j,:] = np.load(cd_base+'/'+list_files[i]+'.npz')['X']
            j+=1
        scaler.partial_fit(X)
        index += partial_size
        if (index/n)*100 > ten_step:
            print(str(ten_step)+'%')
            ten_step+=10
            
    return scaler


class Data_to_use:
    
    """
    Class necessary to create the list of examples to train the model
    """
    
    def __init__(self,bank_directory,directories,datasets,components,input_length):
        
        """
        Initialization
        Parameters
        ----------
            directory: bank directory
            directories: list of ['/TRAINING','/VALIDATION','/TESTING']
            datasets: List of dataset used
            components: list of components considered
            input_length: input_length of one example
        """
        self.bank_directory = bank_directory
        self.directories=directories
        self.components = components
        self.datasets=datasets
        self.input_length=input_length
         
    def create_txt_folder(self):
        """
        Returns
        ----------
            Create the folder of the model that will contain the txt files
        """
        def remove_existing(save_folder):
            """
            Returns
            ----------
                Remove prexisting txt list files
            """
   
            for dire in self.directories:
                file=save_folder+'/Indexes'+dire.split('/')[1]+'.txt'
                if os.path.isfile(file):
                    os.remove(file) 
            return
              
    
        def create_txt_list_files(save_folder):   
            """
            Create the txt of all examples
    
            Parameters
            ----------
                save_folder: folder where you want to save the txt file
       
            Returns
            ----------
                The txt file containing all the examples
   
            """
        
            for dataset in self.datasets:
                load_path = self.bank_directory+dataset
                for dire in self.directories:
                    file=load_path+'/Indexes'+dire.split('/')[1]+'.txt'
                    f = open(self.bank_directory+save_folder+'/Indexes'+dire.split('/')[1]+'.txt', "w")
                    with open(file) as FileObj:
                        for lines in FileObj:
                            if lines.split(',')[1] in self.components:
                                f.write(lines)  
            for dire in self.directories:  
                shuffle_txt(self.bank_directory+save_folder+'/Indexes'+dire.split('/')[1]+'.txt')
            return 
        
        save_folder=self.datasets[0]
        comp=self.components[0]
        for c in self.components[1:]:
            comp=comp+'_'+c
        for i in range(1,len(self.datasets)):
            save_folder=save_folder+'_' +self.datasets[i]
        save_folder=save_folder+'_'+comp+'_'+str(self.input_length)
        
        if os.path.isdir(self.bank_directory+save_folder):
            remove_existing(self.bank_directory+save_folder)
        else:
            os.mkdir(self.bank_directory+save_folder)
        
        create_txt_list_files(save_folder)
        
        return save_folder
     

    def create_list_files(self,save_folder):
        """
        Create the list needed by the generator from a txt file

        Parameters
        ----------
            save_folder: folder where the txt files are saved
        
        Returns
        ----------
            list_filesTot: list of lists with len=len(directories), where each nested list is the list of examples needed to build the generator
        """
        
        def create_list_files(path_table):
            """
            Parameters
            ----------
                path_table: path of the txt file
            """
            
            with open(path_table) as FileObj:
                list_files=[]
                for lines in FileObj:
                    lines=lines.rstrip('\n')
                    list_files.append(lines.split(',')[0]+'/Generators/'+lines.split(',')[1]+
                          '/'+lines.split(',')[2]+'/'+lines.split(',')[3])
        
            return list_files
        
        list_filesTot=[[] for dire in self.directories]
        i=0
        for dire in self.directories:
            path_table=save_folder+'/Indexes'+dire.split('/')[1]+'.txt'
            list_filesTot[i]=create_list_files(path_table)
            print('Number of examples for '+ dire.split('/')[1]+': '+str(len(list_filesTot[i])))
            i+=1
        
        return list_filesTot


def grouping(a,t,thr,dist):
    """
    Parameters
    ----------
       a: vector
       thr: threshold
       dist: distance
       t: time vector
    
    Returns
    ----------
       groups: list of lists where each nested list represents one transient
       
    """
    
    p1 =np.argwhere(np.array(a)>= float(thr))
    eventi=list(p1)
    eventi=[t[ev[0]] for ev in eventi]
    if len(eventi)>0:
        groups = [[eventi[0]]]
        breaks    = [i for i,(a,b) in enumerate(zip(eventi,eventi[1:]),1) if b-a>5]
        groups    = [eventi[s:e] for s,e in zip([0]+breaks,breaks+[None])]
        return groups
    else:
        return []


def count_step_velocity(data_folder_path,soln_folder_path,save_path,components):
    
    """
    This function takes heaviside solutions found by gratsid and save a list of all found steps for different velocity threshold
    
    Parameters
    ----------
       data_folder_path: folder of txt input files
       soln_folder_path: folder of gratsid tables
       save_path: folder where saving
       components: components to use
       
       example:
       data_folder_path = '/Users/giacomo/Documents/PhD/SL_SL_Forecast/Cascadia/Stations_GRATSID/'
       soln_folder_path = cd+'/sols_tables_'
       save_path= cd+'/t_disps_resids'
       components=['E/','N/','U/']
    
    Returns
    ----------
       save a list for each threshold (.pkl file)
    
    """

    #npz files list for all components - equal for all components
    names=id_names_npz(soln_folder_path+components[0])
    
    thresholds=[0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005]
    for thr in thresholds:
        print(thr)
        for c in range(len(components)):
            ##### step list #####
            step_listC=[]

            ############ finally build ############
            soln_folder=soln_folder_path+components[c]
            save_pathc=save_path+'/'+components[c]
            sol_path = soln_folder+names[0]+'.npz'
            options = np.load(sol_path,allow_pickle=True)['options']
            options = options.item()
            data_cols = np.load(sol_path)['data_cols']
         
            ########## Allocate  ##########
            for i in range(len(names)):
                sol_path = soln_folder+names[i]+'.npz'
                data_path = data_folder_path+names[i]+'.txt'
                out_path = save_pathc+names[i]+'.txt'
    
                ## loading in the data and the corresponding gratsid solution table
                data = np.loadtxt(data_path)
                perm = list(np.load(sol_path,allow_pickle=True)['perm'])
                sols = list(np.load(sol_path,allow_pickle=True)['sols'])
    
                ## converting time to python datetime integer and isolating the fit directional components
                t = gen_jjj(data[:,0].astype(int),data[:,1].astype(int),data[:,2].astype(int))
                y = data[:,data_cols] ## columns 3,4,5  (in python indexing) are E,N,U
    
                ## getting the fits
                signal = fit_decompose(t,y,None,options['tik_mul'], \
                sols,np.asarray(perm),options['bigTs'],options['Fs']) 

                ######## getting steps ########
                step_indixes,velocity_step=find_step(signal,t,thr)
                if len(step_indixes)>0:
                    step_indixes=t[step_indixes]
            
                ## allocate into the list  ##
                step_listC.append([names[i],step_indixes])
        
        
            ## save the list ##
            with open(save_path+components[c].split('/')[0]+'_list_steps_'+str(thr)+'.pkl', 'wb') as f:
                pickle.dump(step_listC, f)
    return
    
def add_step(magnitude_steps,weights_GR,n_step,X,Y):

    """
    This functions add n_step steps to Input X and target Y matrices

    Parameters
    ----------
       magnitude_steps: range of possible steps amplitudes
       weights_GR: weights of Gutemberg Richter
       n_step: number of "normal" steps to add
       X: input 
       Y: target (matrix of 0 and 1)
       
    Returns
    ----------
       Save:
        - X: X with the added steps
        - Y: Y with the added steps (matrix of 0 and 1)
        - Y_new: Y with the added steps  (matrix of 0 and 1 and 2,3, where 2 means syntehthic step, 3 smeared step)
    """
    
    ########## possible step amplitudes
    
    
    steps_array=np.linspace(-0.1,0.1,100000)
    
    ########## thereshold step ####################
    thr=0.005 ################################################## FUNDAMENTAL ##5mm (before it was 0.0015)
    steps_array=steps_array[np.where(np.abs(steps_array)>thr)]

    ########## timing of the day
    day_frac = np.array([[i] for i in np.random.uniform(low=0, high=0.8, size=(n_step,))]) #np.random.rand(nsteps,1) to simplify

    ########## grab examples ##########
    # sampling with replacement.
    step_examples=[random.choices(range(X.shape[0]),k=n_step)][0]
    Y_new=Y.copy()
    
    ten_step=10
    for k in range(n_step):
        ## sign of the step ##
        sign=[1 if random.random() < 0.5 else -1][0]
        ## amplitude of the step ##
        step=np.random.choice(magnitude_steps, p=GR)*sign
        ## position of the step in the example ##
        pos_step=[random.sample(range(X.shape[1]), 1)[0]][0]
        ## we add the step ##
        X[step_examples[k],pos_step]=X[step_examples[k],pos_step]+ ((1 - day_frac[k])*step)
        X[step_examples[k],pos_step+1:]=X[step_examples[k],pos_step+1:]+(day_frac[k]*step)
        
        ### if pos_step-1, the 1 is placed when the step starts
        Y[step_examples[k],pos_step-1]=1 
        Y_new[step_examples[k],pos_step-1]=2     
            
        if (k/n_step)*100 > ten_step:
            print(str(ten_step)+'%')
            ten_step+=10
            
    return X,Y,Y_new

