#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  15 16:29:18 2023

@author: giacomo
"""
import sys, os,random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

################## GPU OFF ##################
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")


########################################
# SET A DATASET
input_length=31
cd='/home/giacomo/Documents/S_NEW_U_'+str(input_length)
os.chdir(cd)

datasets=['TRAINING','VALIDATION','TESTING']
XTv=np.load(cd+'/'+datasets[1]+'_tensor_step.npz')['X'] 
YTv=np.load(cd+'/'+datasets[1]+'_tensor_step.npz')['H'] 
YTvg=np.load(cd+'/'+datasets[1]+'_tensor_gratsid.npz')['Y']

#########################################################
# ADD ADDITIONAL STEPS
add_step_flag=False
#########################################################


if add_step_flag==True:

    sys.path.append('/home/giacomo/Documents/Denoiser_GPS/Denoiser_code/')
    from funcs_4_DL_resids import add_step
    from create_time_series_plausible import gutenberg_richter_law
    import range_parameters_for_generating_time_series

    ''' 
    #Set constants for the GR law
    a = 5.0
    b = 1.0
    min_step_size, max_step_size=0.005,0.1 # Maximum and minimum step size (e.g Metres) 
    magnitude_steps = np.linspace(min_step_size, max_step_size, 10000)
    '''
    
    n_values = gutenberg_richter_law(magnitude_steps, a, b)
    weights_GR=[i/sum(n_values) for i in (n_values)]
    
    XTr=np.load(cd+'/'+datasets[0]+'_tensor_step.npz')['X'] 
    YTr=np.load(cd+'/'+datasets[0]+'_tensor_step.npz')['H']
    print('Number of examples with at least one step inside: '+ str(len(np.array(np.argwhere(YTr==1))[:,0])))

    # Number of steps that you want to add
    n_step=20000000
    XTr,YTr,Y_new=add_step(magnitude_steps,weights_GR,n_step,XTr,YTr)
    print(str(n_step)+' have been added')
    print('Number of examples with at least one step inside: '+ str(len(np.array(np.argwhere(YTr==1))[:,0])))
    np.savez(file=cd+'/'+datasets[0]+'_tensor_step_AUGUMENTED.npz',X=XTr,H=YTr) 
else:
    XTr=np.load(cd+'/'+datasets[0]+'_tensor_step_AUGUMENTED.npz')['X'] 
    YTr=np.load(cd+'/'+datasets[0]+'_tensor_step_AUGUMENTED.npz')['H']

#########################################################

batch_size=256

####### scaling
scaling='Mean' #######
if scaling=='Mean':
    scaler_name='Scaler'
    mean=XTr.mean()
    std=XTr.std()

    ######################################################### DA ELIMINARE #########################################################
    #cd='/home/giacomo/Documents/Synthetic_dataset_'+str(XTr[0].shape[0])
    ### load mean and std
    #mean=float(np.load(cd+'/Mean_std.npz')['mean'])
    #std=float(np.load(cd+'/Mean_std.npz')['std'])
    ################################################################################################################################
        
    for jj in range(XTr.shape[0]):
        XTr[jj,:]=(XTr[jj,:]-mean)/std
        if jj<XTv.shape[0]:
            XTv[jj,:]=(XTv[jj,:]-mean)/std
    
if scaling=='MaxMin':
    Min=XTr.min()
    Max=XTr.max()
    XTr=(XTr-Min)/(Max-Min)
    XTv=(XTv-Min)/(Max-Min)
    Yteg=(Yteg-Min)/(Max-Min)
    

print('Training shape: ',XTr.shape,YTr.shape)
print('Validation shape: ',XTv.shape,YTv.shape)

#YTr=YTr.astype(int)
#YTv=YTv.astype(int)
training_generator = tf.data.Dataset.from_tensor_slices((XTr, YTr))
validation_generator = tf.data.Dataset.from_tensor_slices((XTv, YTv))
training_generator=training_generator.batch(batch_size)
validation_generator=validation_generator.batch(batch_size)

from tensorflow import keras
from tensorflow.keras.layers import Reshape,Dense,Bidirectional,LSTM,Flatten,Conv1D,Dropout,Conv1DTranspose,BatchNormalization,Input, concatenate,Masking 
from tensorflow.keras import layers
from tensorflow.keras.models import Model

######## Dense model########
def dense(input_length,target_size=1,dropout=None,dropout_value=None):
    act=tf.nn.leaky_relu
    act='relu'
    #act='relu'
    model = keras.Sequential()

    model.add(Masking(mask_value=np.nan, input_shape=(input_length, 1)))
    model.add(Flatten())
    #model.add(Flatten(input_shape=(input_length,1)))
    model.add(BatchNormalization())
    # model.add(Reshape((input_length),input_shape=(input_length,1)))
    model.add(Dense(100,activation=act))
    model.add(Dense(200,activation=act))
    model.add(Dense(200,activation=act))
    model.add(Dense(200,activation=act))
    #model.add(Dense(200,activation=act))
    #model.add(Dense(200,activation=act))
    if dropout==True:
         model.add(Dropout(dropout_value))
    model.add(Dense(input_length,activation="sigmoid"))
    model.summary()
    return model

def Autoencoder(input_length):
    model = keras.Sequential()
    act=tf.nn.leaky_relu

    # Adding a Reshape layer with mask
    model.add(Reshape(((input_length, 1)), input_shape=(input_length, 1), mask_value=np.nan))
    #model.add(Reshape(((input_length,1)),input_shape=(input_length,1)))

    model.add(Conv1D(filters=64, kernel_size=5, padding="same", strides=3, activation=act))
    model.add(Conv1D(filters=32, kernel_size=5, padding="same", strides=3, activation=act))
    model.add(Conv1D(filters=16, kernel_size=5, padding="same", strides=3, activation=act))
    model.add(Conv1D(filters=16, kernel_size=5, padding="same", strides=3, activation=act))
    model.add(Dropout(rate=0.1))
    model.add(Conv1DTranspose(filters=16, kernel_size=5, padding="same", strides=2, activation=act))
    model.add(Conv1DTranspose(filters=32, kernel_size=5, padding="same", strides=3, activation=act))
    model.add(Conv1DTranspose(filters=64, kernel_size=5, padding="same", strides=3, activation=act))
    model.add(Conv1DTranspose(filters=input_length, kernel_size=5 ,padding="same",activation=act))
    model.add(Flatten())
    model.add(Dense(input_length,activation='sigmoid'))
    model.summary()
    return model

def Autoencoder_skip(input_length):
    
    act=tf.nn.leaky_relu

    # Input layer with a mask
    input_net = Input((input_length, 1), mask_value=np.nan)
    #input_net = Input((input_length,1))

    conv1=Conv1D(filters=64, kernel_size=5, padding="same", strides=3, activation=act)(input_net)
    conv2=Conv1D(filters=32, kernel_size=5, padding="same", strides=3, activation=act)(conv1)
    conv3=Conv1D(filters=16, kernel_size=5, padding="same", strides=3, activation=act)(conv2)
    conv4=Conv1D(filters=16, kernel_size=5, padding="same", strides=3, activation=act)(conv3)
    drop=Dropout(rate=0.1)(conv4)
    up3=Conv1DTranspose(filters=16, kernel_size=5, padding="same", strides=3, activation=act)(drop)
    merge3 = concatenate([conv3,up3],axis = 1)
    up2=Conv1DTranspose(filters=32, kernel_size=5, padding="same", strides=3, activation=act)(merge3)
    merge2 = concatenate([conv2,up2],axis = 1)
    up1=Conv1DTranspose(filters=64, kernel_size=5, padding="same", strides=3, activation=act)(merge2)
    merge1 = concatenate([conv1,up1],axis = 1)
    f=Flatten()(merge1)
    output_net=Dense(input_length,activation='sigmoid')(f)
    model = Model(inputs = input_net, outputs = output_net)
    model.summary()
    return model

#model = dense(XTr.shape[1],dropout=False,dropout_value=0.1)
model= Autoencoder_skip(XTr.shape[1])

#loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)


earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=4, verbose=0, mode='auto')
earlystopperV = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, verbose=0, mode='auto')


model.compile(optimizer,loss= 'binary_crossentropy',metrics=['accuracy']) #custom_binary_crossentropy

history=model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=100,
    callbacks=[earlystopper,earlystopperV],batch_size=564
    )

####################### SAVE THE MODEL #######################
model.save(cd+'/Step_model_skip')
if history.history['loss'][-1]< 0.0050 and history.history['loss'][-1]< 0.0041:
    print('Daje per il modello')
    model.save(cd+'/Step_model_skip')


#### number of subplot
n=3

def make_plot(XTe,YTe,Yteg,n):
    import random
    from sklearn.metrics import mean_squared_error 

    step_example=np.array(np.argwhere(YTe==1))[:,0]
    print(step_example)
    random_station = random.sample(range(len(step_example)-n), 1)[0]
    random_station= step_example[random_station]

    fig,axes=plt.subplots(n,1,figsize=(15,10))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    ii=random_station
    print(random_station)
    
    for i in range(n):  
        #Apply the scaling
        XX_inputo=XTe[ii,:] #-mean)/std
        YY=YTe[ii,:]
        YYg=Yteg[ii,:]
        
        ## Predict ###
        XX_input=XX_inputo.reshape([1,XX_inputo.shape[0]])
        print(XX_input.shape)
        predictions=model.predict(XX_input)
        predictions=predictions.squeeze()
                    
        ### Errors ####
        rms = round(mean_squared_error(YY, predictions, squared=False),5) 
        ##################### PLOT ############################
        
        axes[i].plot(np.arange(XX_input.shape[1]),predictions,color='y',label='Predicted',linewidth=2)  
        axes[i].plot(np.arange(XX_input.shape[1]),YY,color='r',label='target',linewidth=0.5)   
        axes[i].legend(prop={'size': 6},loc='lower right')
        axes[i].tick_params(axis='both', which='major', labelsize=5)
        axes[i].set_xlim([0,XX_input.shape[1]-1])  
        axes[i].set_ylabel('Target',fontsize=8)  
        axes[i].set_ylim([0,1.1])          

        ax=axes[i].twinx()

        p2=ax.scatter(np.arange(XX_input.shape[1]),XX_input,s=5,color='b',label='True',linewidth=2)
        p2=ax.plot(np.arange(XX_input.shape[1]),np.squeeze(XX_input)-YYg,color='c',label='True',linewidth=2)
        ax.set_ylabel('Displacement',color='b',fontsize=8)
        ax.spines['bottom'].set_color('b')
        ax.yaxis.label.set_color('b')
        ax.tick_params(axis='y', colors='b',labelsize=8)
        
        if any(predictions>0.1):
            location_step=np.argwhere(predictions>0.1)[0][0] 
            ax.plot([location_step,location_step+1],[XX_input[0,location_step],XX_input[0,location_step+1]],color='g',label='True',linewidth=2)

        ii=ii+1
                
    #if save_flag==True:
        #fig.savefig(save_folder+file_name+'_seq',dpi=300)
    return plt.show()

make_plot(XTv,YTv,YTvg,n)
make_plot(XTv,YTv,YTvg,n)
make_plot(XTv,YTv,YTvg,n)
make_plot(XTv,YTv,YTvg,n)
