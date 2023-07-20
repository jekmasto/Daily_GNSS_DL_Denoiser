import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os, glob, sys
import datetime 
from datetime import timedelta

### Input your current directory where you have the gratsid folder
root_dir = input('Enter the root directory folder: ')

### Directory name to search for
directory_name = 'sharing_gratsid_tf_in_development'

# Iterate through the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    for dirname in dirnames:
        if dirname == directory_name:
            # Found the directory
            directory_path = os.path.join(dirpath, dirname)
            print("Directory found:", directory_path)
            break
if not directory_path:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_name}' not found. Be sure you have installed Gratsid")
        
sys.path.append(directory_path)
from gratsid_tf_gpu_functions_SHARED import *
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# Vectorization of time where x=years,y=months,z=days
gen_jjj = np.vectorize(lambda x,y,z: datetime.date.toordinal(datetime.date(x,y,z)))

def apply_DL_filter(time_t,input_time,components,dataN,input_length,position,cd_baseO,verbose=None,step_array=None): 
    
    """
    Apply the DL model to a time series
    
    Parameters
    ----------
       time_t: vectorized time vector
       input_time: datetime vector
       components: list of components
       dataN: array of raw data without gaps (gaps are nan) with a shape [len(time_t),len(components)]
       input_length: length of the window (integer)
       position: position of the target within the input window
       scaler_cd: folder of scaling files
       cd_baseO: folder of models
       verbose: bolean, if True print information 
       step_array: array of steps associated to the time-series (from .para file from INGV)
       
    Returns
    ----------
       new_t: time vector of the filtered time series (datetime)
       d: raw data
       PredictionsT: predicted residuals 

    """
    
    ######### Put nan #########
    dato_interpT=np.zeros([len(time_t),input_length,len(components)])
    dato_interpT.fill(np.nan)
    
    ######### put optional known steps 
    if any(not sublist for sublist in step_array):
        step_flag=False
    else:
        step_flag=True
        step_arrayT=np.zeros([len(time_t),input_length,len(components)])
        # fill with nans
        step_arrayT.fill(np.nan)
        
    ## time indexes to keep ##
    t_keep=np.zeros(len(time_t))
    
    ## maximum number of admitted nans ##
    thr_perc=(input_length/100)*20 
    
    ######### Start the loop #########
    for i in range(dataN.shape[0]-input_length):
        vector=dataN[i:i+input_length,0]
        ## count nans ##
        nan_count = np.isnan(vector).sum() 
        if nan_count < thr_perc: 
            ## if the last value is nan the interpolation is not allowed ##
            if vector[-1]!=np.nan:
                ## Components ##
                for c in range(len(components)):
                    d_interp=dataN[:i+input_length,c]
                    
                    ## Interpolate NaN values ##
                    nan_indices = np.isnan(d_interp)
                    indices = np.arange(len(d_interp))
                    d_interp[nan_indices] = np.interp(indices[nan_indices], indices[~nan_indices], d_interp[~nan_indices])
                    
                    ## take only the last n=input_length values ##
                    dato_interp=d_interp[-input_length:]
                    dato_interpT[position+i,:,c]=dato_interp
                    
                    ###### known steps ######
                    if step_flag!=False:
                        range_t=np.arange(i,i+input_length)
                        matching_elements = [x for x in step_array[c] if x in range_t] 
                        if matching_elements:
                            step_arrayT[matching_elements,c]=1
                    if c==0:
                        t_keep[position+i]=1

    t_keep=t_keep.astype('int') 
    ind_t_to_keep=np.argwhere(t_keep==1)
    
    if len(ind_t_to_keep)==0:
        #raise ValueError('Too many nan, the DL filter can not be applied')
        print('Too many nan, the DL filter can not be applied')
        return None,None,None
    
    else:
        print('Apply the denoiser model')

        PredictionsT= []
        d=[]
        for c in range(len(components)):   
            
            if components[c]=='U':
                cd_base=cd_baseO+'_'+components[c]+'_'+str(input_length)
                #cd_base=cd_baseO+'_E_N_'+str(input_length)
            else:
                cd_base=cd_baseO+'_E_N_'+str(input_length)
            
            ##### Load Mean and Std of the training set #####
            mean=float(np.load(cd_base+'/Mean_std.npz')['mean'])
            std=float(np.load(cd_base+'/Mean_std.npz')['std'])

            ##### Remove Nan #####
            X = dato_interpT[:,:,c][~np.isnan(dato_interpT[:,:,c]).any(axis=1)]  
            ##### Remove Median #####
            X = X - np.repeat(np.nanmedian(X,axis=1)[:,None],X.shape[1],axis=1) 
            ##### Rescale #####
            X=(X-mean)/std
        
            ##### Load trained DL MODELS #####
            modelStep=cd_base+'/models/Step_model_skip'
            modelN=cd_base+'/models/Gratsid_modelN_noskip_'+str(position) 
            try:
                modelStep = keras.models.load_model(modelStep)
                model = keras.models.load_model(modelN)
            except (OSError, ImportError):
                try:
                    modelStep = tf.saved_model.load(modelStep)
                    model = tf.saved_model.load(modelN)
                except:
                    try:
                        modelStep = keras.models.load_model(modelStep+'.h5', compile=False)
                        modelStep.compile(keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
                        model = keras.models.load_model(modelN+'.h5', compile=False)
                        loss_fn = keras.losses.MeanSquaredError()
                        optimizer = keras.optimizers.Adam(learning_rate=7e-4)
                        model.compile(optimizer, loss=loss_fn)
                    except:
                        import json
                        # Load model architecture from JSON
                        with open(cd_base+'/models/Arch_w/Step_model_skip_architecture.json', 'r') as json_file:
                            model_architectureStep = json_file.read()
                        modelStep = keras.models.model_from_json(model_architectureStep)
                        # Load model weights
                        modelStep.load_weights(cd_base+'/models/Arch_w/Step_model_skip_weights.h5')

                        with open(cd_base+'/models/Arch_w/Gratsid_modelN_noskip_'+str(position) +'_architecture.json', 'r') as json_file:
                            model_architectureN = json_file.read()
                        # Build the model from architecture
                        model = keras.models.model_from_json(model_architectureN)
                        # Load model weights
                        model.load_weights(cd_base+'/models/Arch_w/Gratsid_modelN_noskip_'+str(position) +'_weights.h5')

                assert modelStep.layers[-1].output_shape[1]==input_length
                target_size=model.layers[-1].output_shape[1]
                assert target_size==np.array(position).size

            if verbose==True:
                print('The components is: ',str(components[c]))
                print('The input length is: ',str(input_length))
                print('The target_size is: ',str(target_size))
                print('The position is: ',str(position))

                ######## Step predictions ########
                predictions_stepD=modelStep.predict(X,verbose=1)
            else:
                predictions_stepD=modelStep.predict(X,verbose=0)
        
            predictions_stepD=tf.squeeze(predictions_stepD)
            
            ###### put optional known steps ######
            if step_flag==True: 
                if np.any(step_arrayT[:, :, c] == 1): #if step_array[c]
                    ST=step_arrayT[:,:,c][~np.isnan(dato_interpT[:,:,c]).any(axis=1)]  
                    real_st=np.argwhere(ST==1)
                    predictions_stepD=np.array(predictions_stepD)
                    predictions_stepD[real_st[:,0],real_st[:,1]]=1
                    
            New_X=tf.transpose(tf.stack([predictions_stepD,X]),[1,2,0])
    
            ######## Noise predictions ########
            if verbose==True: 
                predictionsD=model.predict(New_X,verbose=1)
            else:
                predictionsD=model.predict(New_X,verbose=0)
    
            predictions=np.squeeze(predictionsD)
            PredictionsT.append(predictions) 
        
            d.append(np.squeeze(dataN[ind_t_to_keep,c]))
    
        d=np.array(d)
        PredictionsT=np.array(PredictionsT)
        new_t=np.squeeze(input_time[ind_t_to_keep])
    
        return new_t,d,PredictionsT

def has_two_numbers_before_hyphen(input_str):
    import re

    """
    To check if the character '-' is inside the input string and is preceded by two numbers (like in '178-90')
    """
    
    pattern = r'\d\d-'  # Regular expression pattern for two numbers followed by '-'
    return bool(re.search(pattern, input_str))

def import_resi(file):

    """
    Import a .raw file from INGV (Serpelloni data)
    
    Parameters
    ----------
       file: .path of the .raw file
       
    Returns
    ----------
       longitudes,latitudes: latitude and longitude of the station
       dfN: dataframe of the time-series
            dfN has these columns=['YYMMDD','years','months','days','E','N','U','decimal_date']
       
    """
    
    years=[]
    months=[]
    days=[]
    decimal_t=[]
    E=[]
    N=[]
    U=[]
    sites=[]
    date_dT=[]
    lines=[]
    
    j=0
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            lines.append(line)

            j+=1
            if line and j>2:
                values = line.split()
                decimal_t.append(float(values[0]))
                date_d=decimal_to_datetime(float(values[0])).date()
                date_dT.append(date_d)
                years.append(date_d.year)
                months.append(date_d.month)
                days.append(date_d.day)
                
                if has_two_numbers_before_hyphen(str(values[1])): 
                    val=str(values[1])
                    E.append(float(val.split('-')[0]))
                    N.append(float(val.split('-')[1]))
                    U.append(float(values[5]))
                else:
                    E.append(float(values[1]))
                    N.append(float(values[2]))
                    U.append(float(values[6]))
                    
    if has_two_numbers_before_hyphen(str(values[1])):
        station = values[9]
        longitudes=float(values[10])
        latitudes=float(values[11])
        
    else:
        station = values[10]
        longitudes=float(values[11])
        latitudes=float(values[12])
    
    E=np.array(E)*0.001
    N=np.array(N)*0.001
    U=np.array(U)*0.001
    y=np.vstack([years,months,days,E,N,U,decimal_t])
                    
    y=np.transpose(y) 
    dfN=pd.DataFrame(y[:],columns=['years','months','days','E','N','U','decimal_date'])
    
    dfN['YYMMDD']=pd.Series(dtype='float64')
    for i in range(len(dfN)):
        dfN.loc[i, 'YYMMDD'] = pd.Timestamp(date_dT[i]).to_pydatetime()
    dfN['YYMMDD']= pd.to_datetime(dfN['YYMMDD']).astype('datetime64[ns]')   
    
    datetime_index = pd.DatetimeIndex(dfN.YYMMDD)
    # Check for duplicates
    assert not datetime_index.duplicated().any(), "Datetime series contains duplicates."
    # Check if all dates are increasing
    assert (datetime_index == datetime_index.sort_values()).all(), "Dates in the datetime series are not in increasing order."
    
    return longitudes,latitudes,dfN

def load_step(file):
    """
    Load a step file
    
    Parameters
    ----------
       file: .para step file from INGV 
             A .para file looks like:
             0BOD_GPS 24 0.1
             49    0.0 50.0 2011.26575 2100.0 ANT
 
    Returns
    ----------
       List_steps: List of n lists, where n is the number of components
    """
    
    vals=[]
    j=0
    componentsT=['E', 'N', 'U']
    List_steps=[[] for _ in range(len(componentsT))]
    with open(file, 'r') as file:
        for line in file:
            line = line.strip()
            j+=1
            if line and j>1:
                values = line.split()               
                if values[0]=='49' or values[0]=='7':
                    List_steps[0].append(decimal_to_datetime(float(values[3])))
                if values[0]=='50' or values[0]=='8':
                    List_steps[1].append(decimal_to_datetime(float(values[3])))
                if values[0]=='51' or values[0]=='9':
                    List_steps[2].append(decimal_to_datetime(float(values[3])))
    return List_steps
    
    
class Station:

    """
    Create a class for a given station
    """

    def __init__(self, name,starting_date,last_date,components,cd):
        """
        
        Parameters
        ----------
           name: name of the station
           starting_date: starting date
           last_date: last_date for the analysis
           components: components to use
         """
        
        self.name = name
        self.starting_date=starting_date
        self.last_date=last_date
        self.components=components
        self.cd=cd

    def importdata(self):
        
        """
        Import the timeseries as a dataframe
        
        Returns
        ----------
            dfs: dataframe of the timeseries 
        """
        
        fname=self.cd+'/'+str(self.name)
        print(fname)
        
        ## Check which rows are inside the investigated time period
        rows_to_keep=[]
        with open(fname) as f:
            for i, line in enumerate(f):
                che_giorno=datetime.datetime.strptime(str(int(line.split(' ')[0].split('.')[0]))+str('-')+str(int(line.split(' ')[1].split('.')[0]))+'-'+str(int(line.split(' ')[2].split('.')[0])), '%Y-%m-%d').date()
                rows_to_keep.append(i)
                if che_giorno >=self.last_date.date():
                    break
            
        dfs = pd.read_csv(fname, 
                 delim_whitespace=True,header=0,on_bad_lines='skip',skiprows = lambda x: x not in rows_to_keep)
        
        ## Remove Nans
        dfs=dfs.dropna(axis=1, how='all')
        
        if len(dfs)>0:
            new_cols=['year','months','days','E','N','U','sig_e(m)',
                'sig_n(m)','sig_u(m)']
            new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
            dfs.rename(new_names_map, axis=1, inplace=True)
            
            ############### transform to dates ###############
            dfs['YYMMDD'] = pd.Series(dtype='float64')
            for i in range(len( dfs['year'])):
                dfs.loc[i, 'YYMMDD'] = datetime.datetime.strptime(str(int(dfs['year'][i]))+str('-')+str(int(dfs['months'][i]))+'-'+str(int(dfs['days'][i])), '%Y-%m-%d').date()
            
            #all dataframe with the same format!!!!   
            dfs['YYMMDD'].astype('datetime64[ns]')

            dfs=dfs[dfs.YYMMDD>self.starting_date.date()]
            
            datetime_index = pd.DatetimeIndex(dfs.YYMMDD)
            # Check for duplicates
            assert not datetime_index.duplicated().any(), "Datetime series contains duplicates."
            # Check if all dates are increasing
            assert (datetime_index == datetime_index.sort_values()).all(), "Dates in the datetime series are not in increasing order."

        return dfs
    
    def apply_gratsid(self,vectorT,data,options,use_known_steps,df_stepsAC=None,df_stepsEC=None):
        """
        Returns the list of avaliable stations for each day

        Parameters
        ----------
          station: name of the station
          vectorT: time_vector (vectorized)
          data: matrix of input data
          options: gratsid options
          use_known_steps: 0 or 1, if 1 use steps
          df_stepsAC: dataframe of artificial steps
          df_stepsEC: dataframe of earthquake steps
    
        Returns
        ----------
          signals: signals decomposed obtained from the trajectory fitting
    
        """
        
        vectorT=np.array(vectorT)
        ### all signals ###
        signals=[] 

        if use_known_steps==1:
            ############### steps of the station ###############
            antennas=df_stepsAC.loc[list(df_stepsAC.index[df_stepsAC['station'] == self.station]) ].YYMMDD
            earthquakes=list(df_stepsEC.loc[list(df_stepsEC.index[df_stepsEC['station'] == self.station])].YYMMDD)
            earthquakesT=list(df_stepsEC.loc[list(df_stepsEC.index[df_stepsEC['station'] == self.station])].tgratsid)
    
            known_steps_in=[]
            for a in antennas:
                known_steps_in.append([a.date().year,a.date().month,a.date().day,0])
        
            q=0
            for e in earthquakes:
                known_steps_in.append([e.date().year,e.date().month,e.date().day,earthquakesT[q]])
                q+=1
        
            known_steps_in=np.array(list(map(list,set(map(tuple,np.array(known_steps_in)))))) #remove duplicates
    
        else:
            known_steps_in=np.array([])
        
        if known_steps_in.size>0:
            if len(known_steps_in.shape) == 1:
                known_steps_in = known_steps_in.reshape(1,len(known_steps_in))
                x_steps = gen_jjj(known_steps_in[:,0].astype(int), \
                      known_steps_in[:,1].astype(int), \
                      known_steps_in[:,2].astype(int))
            known_steps = np.hstack([(x_steps-1)[:,None],known_steps_in[:,-1][:,None]])
        else:
            known_steps = []   

        ############## GRATSID ###############
        perm,sols = gratsid_fit(vectorT,data,None,known_steps,options)
        signal = fit_decompose(vectorT,data,None,options['tik_mul'], \
                           sols,perm,options['bigTs'],options['Fs'])
    
        ############## median of all solutions for all signals ##############  
        for ii in range(len(signal)):
            signals.append(np.nanmedian(np.array(signal[ii][:]),axis=0)) 
        
        return signals

##################### Some Utilities #####################
def id_names_txt(soln_folder_path):
    """
    list of the names of the *txt files inside a folder

    Parameters
    ----------
       soln_folder_path: input folder 
    
    Returns
    ----------
        Output: List of the names of the files
    """
    os.chdir(soln_folder_path)
    names=[]
    for file in glob.glob("*.txt"):
        names.append(file.split('.')[0])
    return sorted(names) 
    
def id_names_raw(soln_folder_path):
    """
    list of the names of the *txt files inside a folder

    Parameters
    ----------
       soln_folder_path: input folder 
    
    Returns
    ----------
        Output: List of the names of the files
    """
    os.chdir(soln_folder_path)
    names=[]
    for file in glob.glob("*.iqrx3"):
        names.append(file.split('.')[0])
    return sorted(names) 
    
def datetime_to_decimal(dt):

    """
    Convert a datetime object to a decimal date
    """
   
    year_start = datetime.datetime(dt.year, 1, 1)
    days = (dt - year_start).days
    #is_leap_year = (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0)
    #if is_leap_year:
    #    days -= 1

    decimal_date = dt.year + days / 365

    return decimal_date


def decimal_to_datetime(decimal_date):
    """
    Convert a decimal date to a datetime object
    """
    year = int(decimal_date)
    day_of_year = round((decimal_date - year) * 365)
    
    #print((decimal_date - year) * 365)
    #is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) 
    #if is_leap_year and day_of_year>59: #59 because only after February you should consider leap year
        #day_of_year += 1
    date = datetime.datetime(year, 1, 1) + timedelta(days=day_of_year)
    return date
    
def transform_vector(vector):
    """
    Transfor a vector of decimal dates to a vector of integers
    """
    transformed_vector = np.zeros_like(vector, dtype=int)
    diff = vector[1:] - vector[:-1]  # Calculate the differences between consecutive values

    current_value = 1
    for i, d in enumerate(diff):
        increment=round(d / 0.00273972) #0.00273972 24 hours in decimal time
        transformed_vector[i + 1] = transformed_vector[i] + increment

    return transformed_vector

def derivative(t,data,step):
    """
    Compute the derivate taking into acount a variable time vector
    
    Parameters
    ----------
       t: time vector
       data: data vector
       step: step based on which you want to calculate the derivative

    Returns
    ----------
       derivative: with len(data)-1
    """
    if len(t)!=len(data):
        raise ValueError('The two variables have a different length')
   
    der=np.zeros(len(t)-step)
    for i in range(step,len(data)):
        der[i-step]=(data[i]- data[i-step])/(t[i]-t[i-step])
    return der
    
def influence_radius(Mw): #Nevada formula (http://geodesy.unr.edu/explanationofplots.php) 
    d = 1.15 #empirical coefficient;
    return 10**(0.43*Mw-0.7)/d

def indices(lst, item): 
    #return duplicates within a list
    return [i for i, x in enumerate(lst) if x == item]  

def find_common_elements_with_indexes(array1, array2):

    """
    finds the common elements between two arrays and returns their corresponding indexes in both arrays
    """

    common_elements = []
    common_indexes_array1 = []
    common_indexes_array2 = []

    for index1, element1 in enumerate(array1):
        for index2, element2 in enumerate(array2):
            if element1 == element2 and element1 not in common_elements:
                common_elements.append(element1)
                common_indexes_array1.append(index1)
                common_indexes_array2.append(index2)

    return common_elements, common_indexes_array1, common_indexes_array2
    
################## Functions to run the code in a loop with multiple stations and make animations ##################

def avaliable_stations_vel(soln_folder_path,list_stations,t,new_cols):
    """
    Returns the list of avaliable stations for each day

    Parameters
    ----------
       soln_folder_path: input folder 
       list_stations: list of stations
       t: datetime range
       new_cols: list of columns of the input txt files (e.g. ['YYMMDD','E','DL_E','EMV_E','GrAtSiD_E','N','DL_N','EMV_N','GrAtSiD_N'])
    
    Returns
    ----------
       List of avaliable stations for each day
    """
    n_stations=np.zeros(len(t))
    dft = pd.DataFrame(data=np.column_stack((t,n_stations)),columns=['YYMMDD','Count'])
    dft['YYMMDD']=dft['YYMMDD'].astype('datetime64[ns]')
    listT=list(dft['YYMMDD'])
    Stations=[[] for _ in range(len(listT))] #list of length(t) built by empy lists
    k=0
    ten_step=10
    
    for station in list_stations: #random_stations
        dfs = pd.read_csv(soln_folder_path+'/'+str(station)+'.txt', 
                 delim_whitespace=True,header=0,on_bad_lines='skip')
        dfs=dfs.dropna()
        new_cols=new_cols
        new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
        dfs.rename(new_names_map, axis=1, inplace=True)
        #transform to the same datetime format!!
        dfs['YYMMDD']=dfs['YYMMDD'].astype('datetime64[ns]')
    
        listdfs=list(dfs['YYMMDD'])
        both = set(listT).intersection(listdfs ) #datetime elements in common
        indices_A =[listT.index(x) for x in both]
        indices_B =[listdfs.index(x) for x in both]
        for i,j in zip(indices_A,indices_B):
            Stations[i].append([station,j]) #station and index 
        
        dft.loc[indices_A, 'Count']+=1
        k+=1
    
        perc=(k*100)/len(list_stations)
        if perc>ten_step and ((k-2)*100)/len(list_stations)<ten_step:
            print(str(round(perc))+'%')
            ten_step+=10
        
    return Stations,dft

def avaliable_stations(soln_folder_path,list_stations,t):
    
    """
    Returns the list of avaliable stations for each day, having as input NEVADA files
    columns of a NEVADA file: ['site','YYMMMDD','yyyy.yyyy','__MJD','week','d','reflon',
                '_e0','E','n0(m)','N','u0(m)','U','_ant(m)','sig_e(m)',
                'sig_n(m)','sig_u(m)','__corr_en','__corr_eu','__corr_nu']

    Parameters
    ----------
       soln_folder_path: input folder 
       list_stations: list of stations
       t: datetime range
    
    Returns
    ----------
       List of avaliable stations for each day
    """
    n_stations=np.zeros(len(t))
    dft = pd.DataFrame(data=np.column_stack((t,n_stations)),columns=['YYMMDD','Count'])
    dft['YYMMDD']=dft['YYMMDD'].astype('datetime64[ns]')
    listT=list(dft['YYMMDD'])
    Stations=[[] for _ in range(len(listT))] #list of length(t) built by empy lists
    k=0
    ten_step=10
    
    for station in list_stations: #random_stations
        dfs = pd.read_csv(soln_folder_path+'/'+str(station)+'.txt', 
                 delim_whitespace=True,header=0,on_bad_lines='skip')
        dfs=dfs.dropna()
        new_cols=['year','months','days','E','N','U','sig_e(m)','sig_n(m)','sig_u(m)']
        new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
        dfs.rename(new_names_map, axis=1, inplace=True)
        #df.to_csv(cd+'/'+str(station)+'.txt', index=False,sep=',') 
        
        dfs['YYMMDD'] = pd.Series(dtype='float64')
        for i in range(len( dfs['year'])):
            dfs.loc[i, 'YYMMDD'] = datetime.strptime(str(int(dfs['year'][i]))+str('-')+str(int(dfs['months'][i]))+'-'+str(int(dfs['days'][i])), '%Y-%m-%d').date()

        dfs['YYMMDD']=dfs['YYMMDD'].astype('datetime64[ns]')
    
        listdfs=list(dfs['YYMMDD'])
        both = set(listT).intersection(listdfs ) #datetime elements in common
        indices_A =[listT.index(x) for x in both]
        indices_B =[listdfs.index(x) for x in both]
        for i,j in zip(indices_A,indices_B):
            Stations[i].append([station,j]) #station and index 
        
        dft.loc[indices_A, 'Count']+=1
        k+=1
    
        perc=(k*100)/len(list_stations)
        if perc>ten_step and ((k-2)*100)/len(list_stations)<ten_step:
            print(str(round(perc))+'%')
            ten_step+=10
        
    return Stations,dft
    
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
       
def compute_derivative(soln_folder_path,step,new_cols,save_folder):
    """
    Save as new csv the derivative of time-series  

    Parameters
    ----------
       soln_folder_path: folder of input csv 
       step: step (in days) used to compute the derivative
       new_cols: list of columns of the input txt files (e.g. ['YYMMDD','E','DL_E','EMV_E','GrAtSiD_E','N','DL_N','EMV_N','GrAtSiD_N'])
       soln_folder_path: folder where to save the output csv 
    """

    list_stations=id_names_txt(soln_folder_path)
    for station in list_stations: #random_stations
        dfs = pd.read_csv(soln_folder_path+'/'+str(station)+'.txt', 
                 delim_whitespace=True,header=0,on_bad_lines='skip',skiprows=check_rows(soln_folder_path+'/'+str(station)+'.txt',new_cols))
        dfs=dfs.dropna()

        new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
        dfs.rename(new_names_map, axis=1, inplace=True)
        #transform to the same datetime format!!
        dfs['YYMMDD']=dfs['YYMMDD'].astype('datetime64[ns]')

        datetime_index = pd.DatetimeIndex(dfs.YYMMDD)
        # Check for duplicates
        assert not datetime_index.duplicated().any(), "Datetime series contains duplicates."
        # Check if all dates are increasing
        assert (datetime_index == datetime_index.sort_values()).all(), "Dates in the datetime series are not in increasing order."
 
        y=dfs.values[:,1:]
        y_vel=np.zeros([y.shape[0]-step,y.shape[1]])
    
        ###### Derivative ######
        # time vector
        df_dates = dfs['YYMMDD'].apply(lambda x: x.date())
        days_ago=df_dates[0]-df_dates
        days_ago_as_int=[abs(da.days) for da in days_ago] 
    
        for k in range(y.shape[1]):
            y_vel[:,k]=derivative(days_ago_as_int,y[:,k],step)
        
        #y=np.diff(y,axis=0)
        dfvel = pd.DataFrame(y_vel[:] , columns=new_cols[1:])
        dfvel['YYMMDD']  = pd.Series(dtype='float64') 
        assert len(dfvel['YYMMDD'])==len(dfs['YYMMDD'][step:])
        dfvel['YYMMDD']=list(dfs['YYMMDD'][step:])
        dfvel['YYMMDD']= pd.to_datetime(dfvel['YYMMDD']).astype('datetime64[ns]')
        #date columns as first
        my_column = dfvel.pop('YYMMDD')
        dfvel.insert(0, my_column.name, my_column) 
        dfvel.to_csv(save_folder+'/'+str(station)+'.txt', header=None, index=None, sep=' ', mode='a')
    
    return print('finished')

def conversion_Nevada(cd,cd_saving):
    
    """
    Change a txt file download from Nevad to a format that can be handled

    Parameters
    ----------
       cd: input folder 
       cd_saving: folder of the created new files
       
    Returns
    ----------
       txt files for each input file present inside the input folder
    """
    
    split_s=[0,2,5,7] #indexes where you split the string date (e.g. 08AUG22)
    #months=list(np.linspace(1,12,12,dtype=int))
    monthsN=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

    list_stations=id_names_txt(cd+'/Stations')

    for station in list_stations:
        dfs = pd.read_csv(cd+'/Stations/'+str(station)+'.txt', 
                 delim_whitespace=True,header=0,on_bad_lines='skip')
        dfs=dfs.dropna(axis=1, how='all')
        new_cols=['site','YYMMMDD','yyyy.yyyy','__MJD','week','d','reflon',
                '_e0','E','n0(m)','N','u0(m)','U','_ant(m)','sig_e(m)',
                'sig_n(m)','sig_u(m)','__corr_en','__corr_eu','__corr_nu']
        new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
        dfs.rename(new_names_map, axis=1, inplace=True) 
        Columns=['year','month','day','E','N','U','sig_e','sig_n','sig_u']
        zero_data = np.zeros(shape=(len(dfs['YYMMMDD']),len(Columns)))
        zero_data = pd.DataFrame(zero_data, columns=Columns)
        dates=dfs['YYMMMDD']
        j=0
        for d in dates:
            splittata = []
            for i in range(len(split_s)-1):
                splittata.append(d[split_s[i] :split_s[i+1]])
            if int(splittata[0][0])!=9:
                splittata[0]=int('20'+splittata[0])
            else:
                splittata[0]=int('19'+splittata[0])
            zero_data['year'][j]=splittata[0]
            zero_data['month'][j]=int(monthsN.index(splittata[1])+1)
            zero_data['day'][j]=int(splittata[2])
            zero_data['E'][j]=dfs['E'][j]
            zero_data['N'][j]=dfs['N'][j]
            zero_data['U'][j]=dfs['U'][j]
            zero_data['sig_e'][j]=dfs['sig_e(m)'][j]
            zero_data['sig_n'][j]=dfs['sig_n(m)'][j]
            zero_data['sig_u'][j]=dfs['sig_u(m)'][j]
            j+=1
        zero_data.to_csv(cd_saving+str(station)+'.txt', header=None, index=None, sep=' ', mode='a')
    return print('Finished')

###################### Some other filtering methods ######################
def ema(data, window):
    """
    Compute the exponential moving average causally
    
    Parameters
    ----------
       values: input array
       window: window
    """
    alpha = 2 / (window + 1)
    ema = [data[0]]  # Start with the first data point as the initial EMA value

    for i in range(1, len(data)):
        ema_value = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))
        ema.append(ema_value)

    return np.array(ema)
    
def exp_weighted_moving_av_with_shift(y_in,expo,window_size,shift): 
    
    ### Note, "shift" should always be less than window_size/2

    """
    Exponential moving average

    Parameters
    ----------
       y_in: iput data 
       expo: base of the exponential weighting (if 1 --> linear)
       window_size: size of the moving window
       shift: location of the peak of the weights (in samples -- if shift=2 the peak will be at the sample [window_size-2])
    Returns
    ----------
        y_out: filtered time-series
    """  

    x = np.arange(window_size)
    w = x**expo
    if shift>0: ### Note, "shift" should always be less than window_size/2
        w_up = w[0:-shift]
        w_down = w_up[::-1][1:1+shift]
        w = np.concatenate([w_up,w_down])
        
    w = w/w.sum()
    weighted_disps = []

    for i in range(y_in.size-window_size+1):
        ### Maybe some nans in the time series: in that case, we need to re-normalize the weighting.
        w_incase_nan = w.copy()
        w_incase_nan[np.isnan(y_in[i:i+window_size])] = np.nan
        w_incase_nan = w_incase_nan/w_incase_nan.sum()
        
        ### Applying the weighting
        weighted_disps.append(np.nansum(np.multiply(w_incase_nan,y_in[i:i+window_size])))
    
    ## turning list into numpy array
    weighted_disps = np.array(weighted_disps)
    
    ### There will be nans at the start and nans at the end of the time series if shift>0
    y_out = np.zeros(y_in.size)
    y_out.fill(np.nan)
    y_out[(window_size-1-shift):(window_size-1-shift)+weighted_disps.size] = weighted_disps
           
    return y_out

def hampel_filter(data, window_size=None, n_sigma=None,threshold = None):
    """
    Apply Hampel filter to remove outliers from data.
    
    Parameters
    ----------
       data: numpy array or list, the input data
       window_size: int, the size of the moving window (default: 3)
       n_sigma: int, the number of standard deviations to define the outlier threshold (default: 3)
    
    Returns
    ----------
       filtered_data: numpy array, the filtered data with outliers removed
       outlier_indices: numpy array, the indices of the outliers in the original data
       non_outlier_indices: numpy array, the indices of the non-outliers in the original data
    
    """
    filtered_data = np.array(data)
    outlier_indices = []
    non_outlier_indices = []
    
    # Apply the Hampel filter
    for i in range(len(filtered_data)):
        # Define the window indices
        start = max(0, i - window_size)
        end = min(len(filtered_data), i + window_size + 1)
        
        # Compute the median absolute deviation for the window
        window_mad = np.median(np.abs(filtered_data[start:end] - np.median(filtered_data[start:end])))
        
        # Check for outliers
        if np.abs(filtered_data[i] - np.median(filtered_data[start:end])) > n_sigma * window_mad *threshold:
            outlier_indices.append(i)
        else:
            non_outlier_indices.append(i)
    
    # Remove the outliers (NaN values)
    filtered_data[outlier_indices] = np.nan
    filtered_data=np.array(filtered_data).astype('float')
    filtered_data = filtered_data[~np.isnan(filtered_data)]
    
    outlier_indices = np.array(outlier_indices)
    non_outlier_indices = np.array(non_outlier_indices)
    
    return np.array(filtered_data), outlier_indices, non_outlier_indices


def remove_trend(save_folder,new_cols,save_folder_vel):
    from scipy.stats import linregress
    
    """
    Compute and remove a linear trend from a list of time-series

    Parameters
    ----------
       save_folder: input folder 
       new_cols: list of names of the columns of each new txt files
       save_folder_vel: folder where saving new txt files detrended
    
    Returns
    ----------
       Create new txt files removing trend
    """

    list_stations=id_names_txt(save_folder)
    
    for station in list_stations: #random_stations
        station=station.split('.txt')[0]

        ## expected number of columns
        n=len(new_cols)
        dfs = pd.read_csv(save_folder+'/'+str(station)+'.txt', delim_whitespace=True,header=0,on_bad_lines='skip')
        ## skip lines with the wrong number of columns
        dfs = dfs[dfs.apply(lambda row: len(row) == n, axis=1)]
        ## Reset the index if needed
        dfs = dfs.reset_index(drop=True)
        dfs=dfs.dropna()
        
        if len(dfs)>step:
            new_names_map = {dfs.columns[i]:new_cols[i] for i in range(len(new_cols))}
            dfs.rename(new_names_map, axis=1, inplace=True)
            #transform to the same datetime format!!
            dfs['YYMMDD']=dfs['YYMMDD'].astype('datetime64[ns]')
            datetime_index = pd.DatetimeIndex(dfs.YYMMDD)
            
            # Check for duplicates
            print(station)
            assert not datetime_index.duplicated().any(), "Datetime series contains duplicates."
            # Check if all dates are increasing
            assert (datetime_index == datetime_index.sort_values()).all(), "Dates in the datetime series are not in increasing order."
   
            y=dfs.values[:,1:]
            y_vel=np.zeros([y.shape[0],y.shape[1]])
    
            ###### Derivative ######
            # time vector
            df_dates = dfs['YYMMDD'].apply(lambda x: x.date())
            days_ago=df_dates[0]-df_dates
            days_ago_as_int=np.array([abs(da.days) for da in days_ago]).astype('int')
           
            for k in range(y.shape[1]):
                yy=np.array(y[:,k]).astype('float')
                trend=linregress(days_ago_as_int,yy)
                trend_vector=days_ago_as_int*trend.slope+trend.intercept
                y_vel[:,k]=yy-trend_vector
    
            #y=np.diff(y,axis=0)
            dfvel = pd.DataFrame(y_vel[:] , columns=new_cols[1:])
            dfvel['YYMMDD']  = pd.Series(dtype='float64') 
            dfvel['YYMMDD']=list(dfs['YYMMDD'])[:]
            dfvel['YYMMDD']= pd.to_datetime(dfvel['YYMMDD']).astype('datetime64[ns]')
        
            #date columns as first
            my_column = dfvel.pop('YYMMDD')
            dfvel.insert(0, my_column.name, my_column) 
            dfvel.to_csv(save_folder_vel+'/'+str(station)+'.txt', header=None, index=None, sep=' ', mode='a')
            
            datetime_index = pd.DatetimeIndex(dfvel.YYMMDD)
            # Check for duplicates
            assert not datetime_index.duplicated().any(), "Datetime series contains duplicates."

            # Check if all dates are increasing
            assert (datetime_index == datetime_index.sort_values()).all(), "Dates in the datetime series are not in increasing order."
      
    return print('Finished')

def create_step_file(cd,file_coord,file_step,earthquakes_file,save_Flag=False):

    """
    Create step files to be used by GrAtSiD
    
    Parameters
    ----------
        cd: input directory
        file_coord: file coordinates
        file_step: file of steps (download it from Nevada)
        earthquakes_file: file with all earthquakes downloadable from USGS
        save_Flag: if True save the step and the earthquake file

    Returns
    ----------
        df_stepsAC: dataframe of artificial steps with columns ['station','YYMMDD','number','kind']
        df_stepsEC: dataframe of earthquake-related steps with columns ['station','YYMMDD','number','kind','distance1','distance2','Mw','event_ID','tgratsid']
    """
    
    #import list of all stations
    df = pd.read_csv(file_coord, delimiter=',',names=['station','latitude','longitude'],header=None)

   #split steps in a file with antenna changes and another with earthquakes-related steps
    with open(file_step) as f:
        contents = f.read()

    output1, output2 = contents.split('\n\n')

    data_list1 = output1.split('\n')
    data_list1 = [line.split() for line in data_list1]
    df_stepsA = pd.DataFrame(data_list1, columns=['station','YYMMMDD','number','Change_type'])

    data_list2 = output2.split('\n')
    data_list2 = [line.split() for line in data_list2]
    df_stepsE = pd.DataFrame(data_list2, columns=['station','YYMMMDD','number','distance1','distance2','Mw','event_ID'])

    #pick steps of stations employed in the study
    df_stepsAC=df_stepsA[np.isin(df_stepsA['station'].to_numpy(), list(df.station))]
    df_stepsEC=df_stepsE[np.isin(df_stepsE['station'].to_numpy(), list(df.station))]
    dates=df_stepsEC['YYMMMDD']
    split_s=[0,2,5,7] #indexes where you split the string date (e.g. 08AUG22)
    monthsN=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    j=0
    for d in dates:
        splittata = []
        for i in range(len(split_s)-1):
            splittata.append(d[split_s[i] :split_s[i+1]])
        if int(splittata[0][0])!=9:
            splittata[0]=int('20'+splittata[0])
        else:
            splittata[0]=int('19'+splittata[0])
        df_stepsEC['YYMMMDD']=datetime.datetime.strptime(str(int(splittata[0]))+str('-')+str(int(monthsN.index(splittata[1])+1))+'-'+str(int(splittata[2])), '%Y-%m-%d').date()

    dates=df_stepsAC['YYMMMDD']
    j=0
    for d in dates:
        splittata = []
        for i in range(len(split_s)-1):
            splittata.append(d[split_s[i] :split_s[i+1]])
        if int(splittata[0][0])!=9:
            splittata[0]=int('20'+splittata[0])
        else:
            splittata[0]=int('19'+splittata[0])
        df_stepsAC['YYMMMDD']=datetime.datetime.strptime(str(int(splittata[0]))+str('-')+str(int(monthsN.index(splittata[1])+1))+'-'+str(int(splittata[2])), '%Y-%m-%d').date()
        
    if save_Flag==True:
        df_stepsAC.to_csv(cd+'/Steps_antennas.txt', header=None, index=None, sep=' ', mode='a')
        df_stepsEC.to_csv(cd+'/Steps_earthquakes.txt', header=None, index=None, sep=' ', mode='a')
    
    new_colsAC=['station','YYMMDD','number','kind']
    new_names_map = {df_stepsAC.columns[i]:new_colsAC[i] for i in range(len(new_colsAC))}
    df_stepsAC.rename(new_names_map, axis=1, inplace=True)  
    new_colsEC=['station','YYMMDD','number','distance1','distance2','Mw','event_ID']
    new_names_map = {df_stepsEC.columns[i]:new_colsEC[i] for i in range(len(new_colsEC))}
    df_stepsEC.rename(new_names_map, axis=1, inplace=True)

    ############## Upload steps ##############
    '''
    df_stepsAC = pd.read_csv(cd+'/Steps_antennas.txt', delim_whitespace=True,header=None,
                            on_bad_lines='skip',names=['station','YYMMDD','number','kind'])
    df_stepsEC = pd.read_csv(cd+'/Steps_earthquakes.txt', delim_whitespace=True,header=None,
                            on_bad_lines='skip',names=['station','YYMMDD','number','distance1','distance2','Mw','event_ID'])
    '''
    Min_Mw=float(min(df_stepsEC['Mw'])) #minum magnitude that has caused a step
   
    ############## Import all earthquakes with a magnitude higher than 4.5 form 2008 to 2022 ##############
    E_dfO=pd.read_csv(earthquakes_file, delimiter=',',names=['time','latitude','longitude','depth','mag','magType','nst','gap','dmin','rms','net','id','updated','place','type','horizontalError','depthError','magError','magNst','status','locationSource','magSource'],header=0)
    E_df=E_dfO.copy()
    E_df=E_df.iloc[: , :6] #['time','latitude','longitude','depth','mag','magType']
    E_df['event_ID'] = pd.Series(dtype='string')
    E_df['event_ID']=E_dfO['id']
    E_df=E_df[E_df['mag']>Min_Mw] #all earthquakes with a magnitude higher than Min_Mw
    E_df.reset_index(drop=True, inplace=True) #reset indexes
    E_df['time_gratsid'] = pd.Series(dtype='int')

    ############## Normalize time (in hours) to 1 --> necessary to gratsid ##############
    for i in range(len(E_df)):
        norm=int(E_df['time'][i].split('T')[1].split(':')[0])/24
        E_df.loc[i, 'time_gratsid']=norm

    ############## Take only steps that are insiede the influence radius ##############
    take_indexes=[]
    for i in range(len(df_stepsEC)):
        r=influence_radius(float(df_stepsEC.iloc[i]['Mw']))
        #if r > df_stepsEC.iloc[i]['distance1'] or r> df_stepsEC.iloc[i]['distance2']:
        take_indexes.append(i) #indexes to be considered
    index_level_to_filter = 0
    df_stepsEC[df_stepsEC.index.get_level_values(index_level_to_filter).isin(take_indexes)]
    df_stepsEC.reset_index(drop=True, inplace=True) #reset indexes

    ############## Take only steps that are insiede the influence radius ##############
    take_indexes=[]
    for i in range(len(df_stepsEC)):
        if df_stepsEC.iloc[i]['event_ID'] in list(E_df['event_ID']):
            take_indexes.append(i)

    df_stepsEC[df_stepsEC.index.get_level_values(index_level_to_filter).isin(take_indexes)]
    df_stepsEC.reset_index(drop=True, inplace=True) #reset indexes

    df_stepsEC['tgratsid'] = pd.Series(dtype='float64') # time for gratsid
    for i in range(len(df_stepsEC)):
        for k in range(len( E_df)):
            if df_stepsEC.iloc[i]['event_ID']==E_df.iloc[k]['event_ID']:
                gtime=E_df.iloc[k]['time_gratsid']
                df_stepsEC.loc[i,'tgratsid']= gtime
                
    df_stepsEC[['tgratsid']] = df_stepsEC[['tgratsid']].fillna(0) # time=0 for earthquakes not in the catalog       

    #all dataframe [steps] with the same format!!!! 
    df_stepsAC['YYMMDD'] =  pd.to_datetime(df_stepsAC['YYMMDD']).astype('datetime64[ns]')
    df_stepsEC['YYMMDD'] =  pd.to_datetime(df_stepsEC['YYMMDD']).astype('datetime64[ns]')

    return df_stepsAC,df_stepsEC

def loop_for_apply_filter(cd,cd_base,save_folder,t,components,input_length,position,suffix,cd_step,step_derivative, \
    Remove_outliers_Flag=True,gratsid_flag=False,options_gratsid=None,exp_flag=False,alpha=2.71,Use_steps_flag=True,coordinates_flag=True,coord_file=None):

    """
    Parameters
    ---------
        cd: input folder
        cd_base: folder where you have the models
        save_folder: folder where to save data
        t: time period of investigation
        components: components to use (e.g. ['E','N'])
        input_length: length of the window (integer)
        position: position of the target within the input window
        suffix: suffix of input files (each file is of 1 station)
        cd_step: folder of steps
        step_derivative: temporal step to calculate derivative to extract the trend (in days)
        options_gratsid: options of gratsid
        alpha: base of the exponential to control the weights of the exponential moving average (default=2.71)

        --- Boolean Flags ---
            Remove_outliers_Flag: if True Remove the outliers in a causal way
            gratsid_flag: if True run Gratsid
            exp_flag: if True run exponential moving average
            Use_steps_flag: if True use information of steps
            coordinates_flag: if True, creates a file of coordinates of cleaned stations
        
        coord_file: name of the coordinates file
    """
    componentsT=['E','N','U'] 
    indC = [componentsT.index(comp) for comp in components]
    print('You are using these components: ',components)

    if gratsid_flag==True:
        assert 'options_gratsid' in locals(), "Variable 'options_gratsid' does not exist."

    if suffix=='.txt':
        list_stations=id_names_txt(cd)
    elif suffix=='.resi.cgps_raw.unflt.clean.iqrx3':
        list_stations=id_names_raw(cd)
    else:
        print('check your suffix')

    sss=0
    with open(coord_file, "w") as fileCoord:
        for station in list_stations[:]: 
            
            ##### Reference station #####
            print(station,sss)
    
            ########### import data ###########
            if suffix=='.txt':
                STAT = Station(station+'.txt', t[0],t[-1],components,cd)
                dfsO=STAT.importdata()
            if suffix=='.resi.cgps_raw.unflt.clean.iqrx3':
                file=cd+station+suffix
                longitude,latitude,dfsO =import_resi(file) 
    
            if len(dfsO)>0:
               
                data=dfsO.values
                data_comp=data[:,3+np.array(indC)]

                if data_comp.shape[0]>input_length:
                    
                    if coordinates_flag==True:
                        lineN = station+','+str(latitude)+','+str(longitude)+'\n' 
                        fileCoord.write(lineN)
                    
                    datetime_obj = [datetime.datetime.combine(dfsO['YYMMDD'].iloc[i], datetime.time()) for i in range(len(dfsO['YYMMDD']))] 
                    ## vetorized time
                    t_gO = gen_jjj(data[:,0].astype(int),data[:,1].astype(int),data[:,2].astype(int))
                    t_g=t_gO-t_gO[0]
                    t_g=t_g.astype('int')

                    if Use_steps_flag==True:
                        ########### import Step file ###########
                        file_step=cd_step+station+'.para'
                        if os.path.exists(file_step):
                            List_steps=load_step(file_step)
                            ind_stepT=[[] for _ in range(len(List_steps))]
                            for iii in range(len(List_steps)):
                                for j in range(len(List_steps[iii])):
                                    data_step=datetime.date.toordinal(List_steps[iii][j])-t_gO[0]
                                    if data_step>position:
                                        ind_stepT[iii].append(data_step)
                        if any(sublist for sublist in ind_stepT if ind_stepT):        
                            print('There are some steps')
                            #ind_stepT=[[item for sublist in sublist_list for item in sublist] for sublist_list in ind_stepT]
                    else:
                        ind_stepT=[[] for _ in range(len(components))]
                    ###################### Time - without gaps ######################
                    input_time=np.arange(dfsO['YYMMDD'].iloc[0], dfsO['YYMMDD'].iloc[-1], timedelta(days=1)).astype(datetime.datetime)
                    time_t = np.linspace(t_g[0], t_g[-1], int(t_g[-1]-t_g[0]+1)).astype('int')

                    ###################### Remove Outliers ######################
                    if Remove_outliers_Flag==True:
                        Noutlier_indicesT=[]
                        for c in range(len(components)):
                            filtered_data, outlier_indices, non_outlier_indices=hampel_filter(data_comp[:,c],31, 3,threshold = 3)
                            Noutlier_indicesT.append(non_outlier_indices)

                        Noutlier_indicesT = [item for sublist in Noutlier_indicesT for item in sublist]
                        if len(components)==2:
                            Noutlier_indicesT=list(set([x for x in Noutlier_indicesT if Noutlier_indicesT.count(x) > 1]))
                        elif len(components)==3:
                            Noutlier_indicesT=list(set([x for x in Noutlier_indicesT if Noutlier_indicesT.count(x) > 2]))

                        print('Outliers removed are the: '+str(100-(len(t_g[Noutlier_indicesT])*100)/len(t_g))+' %')

                        t_g=t_g[Noutlier_indicesT]
                        t_g=t_g.astype('int')
                        datetime_obj=list(np.array(datetime_obj)[Noutlier_indicesT])
                        data_comp=data_comp[Noutlier_indicesT,:]

                    ###################### data to denoise (put nan where there is a gap) ######################
                    dataN=np.zeros([len(time_t),len(components)])
                    dataN.fill(np.nan)
                    indexes=[i for i in time_t if i in t_g]
                    dataN[indexes,:]=data_comp

                    ############ DL ############
                    new_t,d,PredictionsT=apply_DL_filter(time_t,input_time,components,dataN,input_length,position,cd_base,verbose=False,step_array=ind_stepT)

                    if new_t is None:
                        print("The station is ignored")
                    else:
                        assert len(new_t)==d.shape[1]
                        denoised=np.array(d.T-PredictionsT.T)

                        time, common_indexes_array1, common_indexes_array2= find_common_elements_with_indexes(datetime_obj, new_t)
                        denoised=denoised[common_indexes_array2,:]

                        y=data_comp.astype('float64') 

                        #################### Gratsid Fit ####################
                        if gratsid_flag==True:
                            if use_known_steps==1:
                                signals=STAT.apply_gratsid(t_g,y,options,use_known_steps,df_stepsAC,df_stepsEC) 
                            else:
                                signals=STAT.apply_gratsid(t_g,y,options,use_known_steps) 

                        filterd=[]
                        for i in range(len(components)):
                            raw=y[:,i]
                            raw=raw[common_indexes_array1] 
                            if gratsid_flag==True:
                                #trajectory model
                                trajectory=(signals[0][:,i]+signals[1][:,i]+signals[2][:,i]+signals[3][:,i]+signals[4][:,i])
                                # Compute the trend as the median of the derivative of the trajectory
                                #trend=np.arange(len(trajectory))*np.median(derivative(t_old,trajectory,step_derivative))+(trajectory[0])[common_indexes_array1]
                                trend=(np.arange(len(trajectory))*np.median(np.diff(trajectory))+(trajectory[0]))[common_indexes_array1]
                                trajectory=trajectory[common_indexes_array1]
                                # Exponential moving average
                                if exp_flag==True:
                                    shift=input_length-position #-1 +1 #beacuse the position is indicated considering that python is 0 based 
                                    EMV=exp_weighted_moving_av_with_shift(y[:,i],alpha,input_length,shift)[common_indexes_array1]

                                    assert len(EMV)==len(trend)==len(trajectory)==len(raw)
                                    # Remove the trend to every products
                                    filterd.append(np.vstack([raw-trend,denoised[:,i]-trend,EMV-trend,trajectory-trend]))
                            else:
                                filterd.append(np.vstack([raw,denoised[:,i]]))

                        input_time=input_time[common_indexes_array1]
                        filterd=np.array(filterd)

                        if filterd.shape[0]==3:
                            filterd=np.vstack([filterd[0,:,:],filterd[1,:,:],filterd[2,:,:]])
                        elif filterd.shape[0]==2:
                            filterd=np.vstack([filterd[0,:,:],filterd[1,:,:]])
                        else:
                            filterd=np.squeeze(filterd)

                        y=np.transpose(filterd) 
                        namesT=[]
                        if gratsid_flag==True and exp_flag==True:  
                            for c in components:
                                namesT.append([c,'DL_'+c,'EMV_'+c,'GrAtSiD_'+c])
                        else:
                            for c in components:
                                namesT.append([c,'DL_'+c])
                        names = [item for sublist in namesT for item in sublist]
                        df_Filterd=pd.DataFrame(y[:],columns=names)

                        ############ SAVE ############
                        df_Filterd['YYMMDD']=pd.Series(dtype='float64')
                        for i in range(len(df_Filterd)):
                            df_Filterd.loc[i, 'YYMMDD'] = pd.Timestamp(time[i]).to_pydatetime()

                        df_Filterd['YYMMDD']= pd.to_datetime(df_Filterd['YYMMDD']).astype('datetime64[ns]')
                        my_column = df_Filterd.pop('YYMMDD')
                        df_Filterd.insert(0, my_column.name, my_column) 

                        # Check for duplicates
                        datetime_index = pd.DatetimeIndex(df_Filterd.YYMMDD)
                        assert not datetime_index.duplicated().any(), "Datetime series contains duplicates."
                        # Check if all dates are increasing
                        assert (datetime_index == datetime_index.sort_values()).all(), "Dates in the datetime series are not in increasing order."

                        df_Filterd.to_csv(save_folder+'/'+str(station)+'.txt', header=None, index=None, sep=' ', mode='a')

                        sss+=1

    if coordinates_flag==False:
        os.remove(coord_file)                   
                               
    return print('Finished')   


