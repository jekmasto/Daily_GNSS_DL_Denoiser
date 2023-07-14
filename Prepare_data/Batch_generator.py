#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  11 15:52:21 2023

@author: giacomo
"""
import numpy as np
import os,keras, glob,sys
from tensorflow.keras.utils import Sequence
sys.path.append('/home/giacomo/Documents/Denoiser_GPS/Denoiser_code/')


class DataGenerator(Sequence):
    """
    Generates data for Keras
   
   Input:
        param bank_directory: bank_directory foleder
        param shuffle: True to shuffle label indexes after every epoch
        dimY: dimension target
        pos: position of target with respect to input (pythonic way_ last element=-1) 
             if you want the central point, pos=int(input_length/2)+1
        """
    def __init__(self, bank_directory=None, dimY=None, pos=None,shuffle=False):
        'Initialization'
        self.bank_directory = bank_directory
        self.dimY = dimY
        self.pos=pos
        self.shuffle = shuffle
        self.on_epoch_end()

    def __id_names_npz(self):
        """
        Return list of the names of the *npz files inside a folder
        
        Parameters
        ----------
           soln_folder_path: input folder 
   
        Returns
        ---------- 
           List of the names of the files
        """
        os.chdir(self.bank_directory)
        names=[]
        for file in glob.glob("*.npz"):
            names.append(file.split('.')[0])
        return sorted(names) 

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.__id_names_npz())/2)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        ID = self.indexes[index:index+1]

        # Generate data
        X, y = self.__data_generation(str(ID[0]))

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.__id_names_npz())/2))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ID):
        """
        Generates data containing batch_size samples
         --> Output:' # X : (n_samples, *dim)
        """
      
        X= np.load(self.bank_directory+'/X' +ID+ '.npz')['X']
        if self.dimY==1:
            y= np.load(self.bank_directory+'/Y' +ID+ '.npz')['Y'][:,self.pos] 
        
        return X, y



input_length=31
cd='/home/giacomo/Documents/S_NEW_U_'+str(input_length)
cd=cd+'/Batch_files'
os.chdir(cd)
target_size=1
position=int(input_length/2)
print('The input_length is: ',input_length,' The position is: ',position)

datasets=['TRAINING','VALIDATION','TESTING']
cd_tr=cd+'/'+datasets[0]
cd_val=cd+'/'+datasets[1]

training_generator = DataGenerator(cd_tr,target_size,position)
validation_generator = DataGenerator(cd_val,target_size,position)
print('Generators have been succesfuly created')


from Best_model import *
#model = Autoencoder((input_length,2)) #lr: 5e-4 loss: 1.6614e-07 - val_loss: 1.0703e-07 -- 2.45e-07
#model=Convolution((input_length,2)) 
#input_shape = (New_XTr[0].shape[0],New_XTr[0].shape[1])
#model = build_modelT(New_XTr[0].shape, head_size=256,num_heads=4, ff_dim=4,num_transformer_blocks=4,mlp_units=[128],mlp_dropout=0.1,dropout=0.1)
#model = dense(New_XTr[0].shape,dropout=False,dropout_value=0.1)
model= model_lstm((input_length,2),target_size=target_size)
model.summary()

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=7e-4) 

model.compile(optimizer,loss=loss_fn)

earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=4, verbose=0, mode='auto')
earlystopperV = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=4, verbose=0, mode='auto')

hystory=model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[earlystopper,earlystopperV],
    ) #batch_size=1


cd='/home/giacomo/Documents/S_NEW_U_'+str(input_length)    
model.save(cd+'/Gratsid_modelN_noskip_'+str(position))


####################### FROM HERE JUST TO PLOT #######################



### load mean and std
mean=float(np.load(cd+'/Mean_std.npz')['mean'])
std=float(np.load(cd+'/Mean_std.npz')['std'])

### load models
try:
    modelStep = keras.models.load_model(cd+'/Step_model_skip')
    model = keras.models.load_model(cd+'/Gratsid_modelN_noskip_'+str(input_length))
except:
    modelStep = tf.saved_model.load(cd+'/Step_model_skip')
    model =tf.saved_model.load(cd+'/Gratsid_modelN_noskip_'+str(input_length))


dataset='/TRAINING'

cd='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset/'
#folder_model='/home/giacomo/Documents/Synthetic_dataset_61/'
folder_model='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset'+str(input_length)


##### Plot n examples #####
n=2
save_flag=False

sys.path.append('/home/giacomo/Documents/Synthetic_dataset/code')
from Combined_make_plot import plot_examples_functions_whole_time_series_indipendent
plot_examples_functions_whole_time_series_indipendent(cd,dataset,folder_model,model,modelStep,position,n,mean,std,save_flag=save_flag) 
plot_examples_functions_whole_time_series_indipendent(cd,dataset,folder_model,model,modelStep,position,n,mean,std,save_flag=save_flag) 
plot_examples_functions_whole_time_series_indipendent(cd,dataset,folder_model,model,modelStep,position,n,mean,std,save_flag=save_flag) 
