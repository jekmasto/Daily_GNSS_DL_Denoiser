"""
Created on Mar Feb 21 19:53:25 2023

@author: jon - giacomo

Create batch files 
"""

import sys, os,random
import numpy as np
import tensorflow as tf
from tensorflow import keras

################## GPU OFF ##################
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape,Dense,Bidirectional,LSTM,Flatten,Conv1D,Dropout,Conv1DTranspose,BatchNormalization
sys.path.append('/home/giacomo/Documents/Denoiser_GPS/Denoiser_code/')

#########################################################
#                    SET A DATASET
input_length=31
cd='/home/giacomo/Documents/S_NEW_U_'+str(input_length)
os.chdir(cd)

#position=int(input_length/2)
print('The input_length is: ',input_length)
#########################################################
batch_size=256

datasets=['TRAINING','VALIDATION','TESTING']

cd_saving=cd+'/Batch_files'
print(cd_saving)
isExist = os.path.exists(cd_saving)
if not isExist:
    os.mkdir(cd_saving)

for d in datasets[:2]:
    print(cd_saving+'/'+d)
    isExist = os.path.exists(cd_saving+'/'+d)
    if not isExist:
        os.mkdir(cd_saving+'/'+d)

XTr=np.load(cd+'/'+datasets[0]+'_tensor_gratsid.npz')['X'][:,:] #10000000
YTr=np.load(cd+'/'+datasets[0]+'_tensor_gratsid.npz')['Y'][:,:] #10000000 #[:,position] 9000000
XTv=np.load(cd+'/'+datasets[1]+'_tensor_gratsid.npz')['X'][:] #-3000000
YTv=np.load(cd+'/'+datasets[1]+'_tensor_gratsid.npz')['Y'][:,:]
#XTe=np.load(cd+'/'+datasets[2]+'_tensor_gratsid.npz')['X'] 
#YTe=np.load(cd+'/'+datasets[2]+'_tensor_gratsid.npz')['Y'] [:,position]

#########################################################
try:
    modelStep = keras.models.load_model(cd+'/Step_model_skip')
except:
    modelStep =tf.saved_model.load(cd+'/Step_model_skip')


scaler_name='Scaler' 
### save mean and std
mean=XTr.mean()
std=XTr.std()
np.savez(file=cd+'/Mean_std',mean=mean,std=std) 
### load mean and std
mean=float(np.load(cd+'/Mean_std.npz')['mean'])
std=float(np.load(cd+'/Mean_std.npz')['std'])
    
################################################################################################################################
ss=0
T_l=XTr.shape[0]
ten_step=5

for kk in range(batch_size,XTr.shape[0],batch_size):
    for jj in range(kk-batch_size,kk):
        XTr[jj,:]=(XTr[jj,:]-mean)/std
        if jj<XTv.shape[0]:
            XTv[jj,:]=(XTv[jj,:]-mean)/std
    if kk<XTv.shape[0]:
        predictionsTv=modelStep.predict(XTv[kk-batch_size:kk,:],verbose=0 )
        predictionsTv=tf.squeeze(predictionsTv) 
        New_XTv=tf.transpose(tf.stack([predictionsTv,XTv[kk-batch_size:kk,:]]),[1,2,0])
        np.savez(file=cd_saving+'/'+datasets[1]+'/X'+str(ss)+'.npz',X=New_XTv) 
        np.savez(file=cd_saving+'/'+datasets[1]+'/Y'+str(ss)+'.npz',Y=YTv[kk-batch_size:kk,:]) 


    predictionsTr=modelStep.predict(XTr[kk-batch_size:kk,:],verbose=0)
    predictionsTr=tf.squeeze(predictionsTr)
    New_XTr=tf.transpose(tf.stack([predictionsTr,XTr[kk-batch_size:kk,:]]),[1,2,0])
    np.savez(file=cd_saving+'/'+datasets[0]+'/X'+str(ss)+'.npz',X=New_XTr) 
    np.savez(file=cd_saving+'/'+datasets[0]+'/Y'+str(ss)+'.npz',Y=YTr[kk-batch_size:kk,:]) 
    ss+=1

    if (kk/T_l)*100 > ten_step:
        print(str(ten_step)+'%'+' of examples processed')
        ten_step+=5
    


