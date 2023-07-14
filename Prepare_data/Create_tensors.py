import sys, os,pickle,warnings,random
import numpy as np
import pandas as pd
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras

################## GPU OFF ##################
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
sys.path.append('/home/giacomo/Documents/Denoiser_GPS/sharing_gratsid_tf_in_development')
sys.path.append('/home/giacomo/Documents/Denoiser_GPS/Denoiser_code/')
from gratsid_tf_gpu_functions_SHARED import *
from funcs_4_DL_resids import id_names_txt,derivative,grouping,id_names_npz,Data_to_use


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
        H_vector: matrix of targets step model
    
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
    
def create_XY(load_path,save_path,components,perctr,percval,directories,write_txt=True):
    import random
    
    """
    Create .npy tensors
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

    #indexes=np.load('/home/giacomo/Documents/Step_model/Stations_indices.npy')
    
    
    indexes=list(range(len(names)))
    random.shuffle(indexes)
    
    assert len(names)==len(indexes)
    names=[names[i] for i in indexes]
    
    
    #### Training-validation-testing split ####
    start=0
    startval=int((len(names)/100)*perctr) 
    endval=int((len(names)/100)*percval)+startval 
    starters=[start,startval,endval,len(names)-1]
    
    #### Loop into the dataset ####
    for d in range(len(directories)):
        print(directories[d].split('/')[1])
        names_new=names[starters[d]:starters[d+1]]
        
        #### Where I will allocate data ####
        XT=[]
        YT=[]
        HT=[]

        isExist = os.path.exists(save_path+'/Generators/')
        if not isExist:
            os.mkdir(save_path+'/Generators/')
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

                #### Load data #### 
                in_path = load_path_component+names_new[i]+'.npz'
                t = np.load(in_path)['t']
                X = np.load(in_path)['X']
                Y = np.load(in_path)['Y']
                H = np.load(in_path)['H'] 

                list_transient=np.load(in_path)['list_transient']

                if X.size==0:
                    print('Urca')
                    warnings.warn('Warning '+load_path_component+names_new[i]+'.npz'+' is empty!!!')
                    os.remove(os.path.join(load_path_component, names_new[i]+'.npz')) ###delete the file
                else:

                    #### no nan indexes
                    nonan_ind=[ind[0] for ind in np.argwhere(~np.isnan(X).any(axis=1))]
                    nonan_indy=[ind[0] for ind in np.argwhere(~np.isnan(Y).any(axis=1))]
                    nonan_indH=[ind[0] for ind in np.argwhere(~np.isnan(H).any(axis=1))]
                    
                    assert nonan_ind==nonan_indy==nonan_indH #n  ###    just to confirm
                    
                    nonanX=X[nonan_ind]
                    nonanY=Y[nonan_indy]
                    nonanH=H[nonan_indH]

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
                        
                        #### Create txt file #### 
                        with open(save_path+'/Indexes'+directories[d].split('/')[1]+'.txt', "a") as file:  
                           file.write(area+','+components[c].split('/')[0]+','+names_new[i]+','+str(ii)+ '\n')   
                        
                        
                        XT.append(nonanX[ii,:])
                        YT.append(nonanY[ii,:])
                        HT.append(nonanH[ii,:])

        #### Shuffle indexes of txt files ####
        path_indexes=save_path+'/Indexes'+directories[d].split('/')[1]+'.txt'
        
        #indices=shuffle_txt_indices(path_indexes)
        indices=shuffle_txt_indices(save_path+'/Indexes_functions'+directories[d].split('/')[1]+'.txt')
        #print(indices)
        np.save(save_path+'/Indici_random'+directories[d].split('/')[1],indices)

        #indices=np.load('/home/giacomo/Documents/Step_model/Indici_random'+directories[d].split('/')[1]+'.npy',allow_pickle=True)
        #indices=[int(i) for i in indices]
        
        #### Shuffle the tensor
        XT=np.asarray(XT)     
        YT=np.asarray(YT)
        HT=np.asarray(HT)
        XT=XT[indices,:]
        YT=YT[indices,:]
        HT=HT[indices,:]
        
        XT=np.squeeze(XT)
        HT=np.squeeze(HT)
        YT=np.squeeze(YT)
        
        print(XT.shape,YT.shape, HT.shape) 

        np.savez(file=save_path+'/'+directories[d].split('/')[1]+'_tensor_step.npz',X=XT,H=HT) 
        np.savez(file=save_path+'/'+directories[d].split('/')[1]+'_tensor_gratsid.npz',X=XT,Y=YT) 

    names=id_names_npz(load_path+'/'+components[0])
    assert len(indexes)==len(names)
    np.save(save_path+'/Stations_indices',indexes)
    
    
    DATASETS=[save_path.split('/')[-1]]
    DATA=Data_to_use(save_path,directories,DATASETS,components,nonanX.shape[0])
    ############### Create list necessary by the generator #################
    list_filesTot=DATA.create_list_files(save_path)
    ##### Save tables as pickle #####
    
    with open(save_path+'/indixes_list_generator_stepModel.pkl', 'wb') as f:
        pickle.dump(list_filesTot, f)
    
        
    return 


# SET A DATASET
cd='/home/giacomo/Documents/S_NEW'
os.chdir(cd)


max_gap = 5 ### Maximum gap for having nans in the series --> for interpolating time series
input_length = 31 #### Length of the time series considered when estimating vels
load_path = cd+'/t_disps_residsF_U' #### path of txt files 

cd_saving=cd+'_U_'+str(input_length)
print(cd_saving)
isExist = os.path.exists(cd_saving)
if not isExist:
    os.mkdir(cd_saving)


save_path = cd_saving+'/tXY_by_component' #### path of npz files to create

############ Set the components ############
components=['U/'] #### Just E, synthetics are not divided in components ####
Make_X_Y(max_gap,input_length,load_path,save_path,components)
print('X_Y created')


directories=['/TRAINING','/VALIDATION','/TESTING']
############ Percentage by which we split the dataset ############
perctr=60 #training
percval=20 #validation

############ path of .npy files to create ############
load_path = cd_saving+'/tXY_by_component'
save_path=cd_saving
create_XY(load_path,save_path,components,perctr,percval,directories,write_txt=True)
