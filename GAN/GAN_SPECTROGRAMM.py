from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random,librosa,glob,os, PIL,time
from numpy.random import rand
from numpy.random import randn
from scipy import signal
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display

################## GPU OFF ##################
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")


def Spectrogram(Time, Data,fs): 
    f, t, Sxx = signal.spectrogram(Data, fs,nperseg=120)
    return f,t,Sxx

cd_save='GAN_SPECTROGRAM'
#################### IMPORT DATA ####################
# SET A DATASET
cd_base='/home/giacomo/Documents/Denoiser_GPS/Wordwide_dataset'
os.chdir(cd_base)
load_path = cd_base+'/t_disps_resids' #### path of txt files
max_gap = 5 ### Maximum gap for having nans in the series --> for interpolating time series

cd_saving=cd_base+'/Residuals_tensors'
print(cd_saving)
isExist = os.path.exists(cd_saving)
if not isExist:
    os.mkdir(cd_saving)
save_path = cd_saving

############ Set the components ############
components=['E/','N/'] ################### solo U 'U/'

############ Build the tensor 
years=3
R_Residuals=np.load(cd_saving+'/E_N_Residuals_'+str(years)+'years.npy')  #U_Residuals
R_Residuals=np.vstack(np.transpose(R_Residuals,[2,0,1]))

### shuffle ###
idx = np.random.permutation(len(R_Residuals))
R_Residuals = np.asarray(R_Residuals)[idx]

############ Random data ############
n=np.array(R_Residuals).shape[1]
Std=np.array(R_Residuals).std()
Mean=np.array(R_Residuals).mean() 

random_input = Std*randn(np.array(R_Residuals).shape[0]* n) + Mean
random_input=np.reshape(random_input,(np.array(R_Residuals).shape[0],np.array(R_Residuals).shape[1]))
print('Real data shape: ',R_Residuals.shape)
print('Random data shape: ',random_input.shape)

#################### Compute PSD ####################
n_ftt_w=95 #95
Sxx=[] #real method 1
SxxI=[] #real for training
S_RT=[] #random 
fs=1
Time=R_Residuals.shape[1]
for jj in range(R_Residuals.shape[0]):
    f,t,Sx=Spectrogram(Time,  R_Residuals[jj,:],fs)
    S = np.abs(librosa.stft(R_Residuals[jj,:],n_fft=n_ftt_w))
    #S=np.expand_dims(S,axis=2)
    #S=np.array(tf.image.resize(S, (S.shape[0]-1,S.shape[1])))
    #S=np.squeeze(S)

    #Sx=np.expand_dims(Sx,axis=2)
    #Sx=np.array(tf.image.resize(Sx, (Sx.shape[0]-1,Sx.shape[1])))
    #Sx=np.squeeze(Sx)

    Sxx.append(Sx)
    SxxI.append(S)
    S_R = np.abs(librosa.stft(random_input[jj,:],n_fft=n_ftt_w))
    #S_R=np.expand_dims(S_R,axis=2)
    #S_R=np.array(tf.image.resize(S_R, (S_R.shape[0]-1,S_R.shape[1])))
    #S_R=np.squeeze(S_R)
    S_RT.append(S_R)
    
    
Sxx=np.array(Sxx)
SxxI=np.array(SxxI)
S_RT=np.array(S_RT)
print(Sxx.shape)
print('Data for training shape: ',SxxI.shape)
print('Random data shape: ',S_RT.shape)

#################### Scale DATA ####################
SxxI=(SxxI-SxxI.min())/(SxxI.max()-SxxI.min())
S_RT=(S_RT-SxxI.min())/(SxxI.max()-SxxI.min())
train_images = SxxI.reshape(SxxI.shape[0], SxxI.shape[1],  SxxI.shape[2], 1).astype('float32')
random_images = S_RT.reshape(S_RT.shape[0], S_RT.shape[1],  S_RT.shape[2], 1).astype('float32')
print('Data for training shape: ',train_images.shape)
print('Random data shape: ',random_images.shape)

#################### Build Dataset ####################
BUFFER_SIZE = 60000
BATCH_SIZE = 64 #1024
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
random_dataset = tf.data.Dataset.from_tensor_slices(random_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#################### Build Model ####################
# Define the generator model
'''
def make_generator_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(6*11*34, use_bias=False, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((6, 11, 34)))
    assert model.output_shape == (None, 6, 11, 34)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(17, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(4, 2), padding='same', use_bias=False, activation='tanh'))

    model.add(layers.ZeroPadding2D(padding=(1, 0)))
    model.summary()
    #model.add(layers.Reshape(( train_images.shape[1], train_images.shape[2], 1)))
    
    assert model.output_shape == (None, train_images.shape[1], train_images.shape[2], 1)
 
    return model
'''
def make_generator_model(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(6*6*192, use_bias=False, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((6, 6, 192)))
    model.add(layers.Conv2DTranspose(96, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(48, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (7, 7), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    #model.add(layers.ZeroPadding2D(padding=(1, 0)))
    model.summary()
    #model.add(layers.Reshape(( train_images.shape[1], train_images.shape[2], 1)))
    
    assert model.output_shape == (None, train_images.shape[1], train_images.shape[2], 1)
 
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[train_images.shape[1], train_images.shape[2], 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(96, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(192, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return real_loss,fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.zeros_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim]) #,mean=train_images.mean(),stddev=train_images.std()

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        real_loss,fake_loss=discriminator_loss(real_output, fake_output)
        disc_loss = real_loss + fake_loss

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return real_loss,fake_loss,gen_loss

#################### Build Generator ####################
input_shape = (10,)
generator = make_generator_model(input_shape)
noise = tf.random.normal([1, input_shape[0]]) #,mean=train_images.mean(),stddev=train_images.std()
generated_image = generator(noise, training=False)
plt.figure()
plt.pcolormesh(tf.squeeze(generated_image[0, :, :, 0]),shading='gouraud',cmap='hot_r')
plt.close()

#################### Build Discriminator ####################
generator_optimizer = tf.keras.optimizers.Adam(5e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

#################### HYPERPARAMETERS ####################
EPOCHS = 500
noise_dim = input_shape[0]
num_examples_to_generate = BATCH_SIZE
# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)

seed = tf.random.normal([num_examples_to_generate, noise_dim]) #,mean=train_images.mean(),stddev=train_images.std()
seed_PCA = tf.random.normal([train_images.shape[0], noise_dim]) #,mean=train_images.mean(),stddev=train_images.std()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
                                 
def train(dataset,random_dataset, epochs,train_images):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            real_loss,fake_loss,gen_loss=train_step(image_batch)
         

        # Save the model every 15 epochs
        if (epoch + 1) % 20 == 0:
            sample=random.sample(range(len(train_images)),k=BATCH_SIZE)
            image_batch=train_images[sample,:,:]
            #print('Real Discriminator loss: ',real_loss.numpy(),'Fake Discriminator loss: ',fake_loss.numpy(),'Generator loss: ',gen_loss.numpy())
            checkpoint.save(file_prefix = checkpoint_prefix)
            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                             epoch + 1,
                             seed,image_batch,random_images,train_images,seed_PCA)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed,image_batch,random_images,train_images,seed_PCA)
    
def generate_and_save_images(model, epoch, test_input,image_batch,random_input_batch,train_images,seed_PCA):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(20, 15))

    for ii in range(16):
        ax=plt.subplot(4, 4, ii+1)
        Sx=tf.squeeze(predictions[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        f,t,Sx=Spectrogram(range(R_Residuals[0,:].shape[0]), reverse,fs)
        im=plt.pcolormesh(t,f,Sx,shading='gouraud',cmap='hot_r',vmin=8e-09,vmax=0.0011)
        ax.tick_params(axis='both',which='major',labelsize=7)
    
    cbar_ax=fig.add_axes([0.93,0.106,0.02,0.775])
    cbar=fig.colorbar(im,cax=cbar_ax)
    cbar.ax.set_ylabel('Power',rotation=270)
    fig.suptitle('Fake Examples',
             fontsize=10)
    plt.savefig('/home/giacomo/Documents/Synthetic_dataset/'+cd_save+'/Fake_image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()  
    plt.close()
    
    fig2 = plt.figure(figsize=(24, 15))
    for ii in range(16):
        ax=plt.subplot(4, 4, ii+1)
        Sx=tf.squeeze(image_batch[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        f,t,Sx=Spectrogram(range(R_Residuals[0,:].shape[0]), reverse,fs)
        im=plt.pcolormesh(t,f,Sx,shading='gouraud',cmap='hot_r',vmin=8e-09,vmax=0.0011)
        ax.tick_params(axis='both',which='major',labelsize=7)
        #print('Max: ',Sx.max(),'Min: ',Sx.min())
        #
        #

    cbar_ax=fig2.add_axes([0.93,0.106,0.02,0.775])
    cbar=fig2.colorbar(im,cax=cbar_ax)
    cbar.ax.set_ylabel('Power',rotation=270)

    fig2.suptitle('Real Examples',
             fontsize=10)
    plt.savefig('/home/giacomo/Documents/Synthetic_dataset/'+cd_save+'/Real_image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()  
    plt.close()
    
    fig3 = plt.figure(figsize=(20, 15))
    for ii in range(16):
        ax=plt.subplot(4, 4, ii+1)
        ### Fake
        Sx=tf.squeeze(predictions[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        plt.scatter(range(reverse.shape[0]),reverse,s=2,c='firebrick',linewidths=0.3,label='Fake')
        ### Real
        Sx=tf.squeeze(image_batch[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        plt.scatter(range(reverse.shape[0]),reverse,s=2,c='navy',linewidths=0.3,label='True')
        plt.legend(loc='upper left')
        ax.set_xlim([0.00001,reverse.shape[0]])
        ax.tick_params(axis='both',which='major',labelsize=7)

    
    plt.savefig('/home/giacomo/Documents/Synthetic_dataset/'+cd_save+'/ts_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.close()

    fig5 = plt.figure(figsize=(20, 15))
    for ii in range(16):
        ax=plt.subplot(4, 4, ii+1)
        ### Fake
        Sx=tf.squeeze(predictions[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        (f, S)= scipy.signal.welch(reverse, fs=1,nperseg=500)
        plt.semilogx(f,S,linewidth=1,color='firebrick',label='Fake')
        ### Real
        Sx=tf.squeeze(image_batch[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        (f, S)= scipy.signal.welch(reverse, fs=1,nperseg=500)
        plt.semilogx(f,S,linewidth=1,color='navy',label='True')
        Sx=tf.squeeze(random_input_batch[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        (f, S)= scipy.signal.welch(reverse, fs=1,nperseg=500)
        plt.semilogx(f,S,linewidth=1,color='green',label='Random')
        plt.legend(loc='upper left')
        ax.set_xlim([0.0001,max(f)])
        ax.tick_params(axis='both',which='major',labelsize=7)
    
    
    plt.savefig('/home/giacomo/Documents/Synthetic_dataset/'+cd_save+'/PSD_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.close()
    
    ############ Cumulative PSD
    SF=0
    SR=0
    SRD=0
    for ii in range(predictions.shape[0]):
        Sx=tf.squeeze(predictions[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        (f, S)= scipy.signal.welch(reverse, fs=1,nperseg=500)
        SF=SF+S
        
        ### Real
        Sx=tf.squeeze(train_images[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        (f, S)= scipy.signal.welch(reverse, fs=1,nperseg=500)
        SR=SR+S

        Sx=tf.squeeze(random_input_batch[ii, :, :, 0])
        Sx=np.array(Sx)
        reverse=librosa.griffinlim(Sx,win_length=n_ftt_w-2)
        (f, S)= scipy.signal.welch(reverse, fs=1,nperseg=500)
        SRD=SRD+S

    fig5, ax = plt.subplots(1,1,figsize=(20,15))
    plt.semilogx(f,SF,linewidth=1,color='firebrick',label='Fake')
    plt.semilogx(f,SR,linewidth=1,color='navy',label='True')
    plt.semilogx(f,SRD,linewidth=1,color='green',label='Random')
    plt.legend(loc='upper left')
    ax.set_xlim([0,max(f)])
    ax.tick_params(axis='both',which='major',labelsize=7)

    plt.savefig('/home/giacomo/Documents/Synthetic_dataset/'+cd_save+'/PSD_CUM_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.close()

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    ############ FOR THE PCA DATA HAS TO BE 2D
    predictions = model(seed_PCA, training=False)
    pred_im=predictions
    fake_loss = cross_entropy(tf.zeros_like (predictions), predictions)
    real_loss = cross_entropy(tf.ones_like(train_images), train_images)
    gen_loss = generator_loss(predictions)
    print('Real Discriminator loss: ',real_loss.numpy(),'Fake Discriminator loss: ',fake_loss.numpy(),'Generator loss: ',gen_loss.numpy())

    predictions=np.array(tf.squeeze(predictions))
    predictions=predictions.reshape((predictions.shape[0],predictions.shape[1]*predictions.shape[2]))

    train_images=np.array(tf.squeeze(train_images))
    train_images=train_images.reshape((train_images.shape[0],train_images.shape[1]*train_images.shape[2]))

    random_input_batch=np.array(tf.squeeze(random_input_batch))
    random_input_batch=random_input_batch.reshape((random_input_batch.shape[0],random_input_batch.shape[1]*random_input_batch.shape[2]))

    #print('Real shape: ',train_images.shape,'Fake shape: ',predictions.shape,'Random shape: ',random_input_batch.shape)
    ############ PCA ############
    n_components = 2
    pca = PCA(n_components=n_components)
    tsne = TSNE(n_components=n_components, n_iter=300,perplexity=30)
    
    #The fit of the methods must be done only using the real sequential data
    pca.fit(train_images)
    pca_real = pd.DataFrame(pca.transform(train_images))
    pca_synth = pd.DataFrame(pca.transform(predictions))
    pca_random= pd.DataFrame(pca.transform(random_input_batch)) 
    
    ############ TSNE ############
    data_reduced = np.concatenate((train_images, predictions,random_input_batch), axis=0)
    tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

    
    #scatter plot
    fig4,axes=plt.subplots(1,2,figsize=(20,15))
    axes[0].set_title('PCA results',
             fontsize=20,
             color='red',
             pad=10)
             
    #PCA scatter plot
    axes[0].scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:,1].values,s=2,
            c='black', alpha=0.2, label='Original')
    axes[0].scatter(pca_synth.iloc[:,0], pca_synth.iloc[:,1],s=2,
            c='red', alpha=0.2, label='Synthetic')
    axes[0].scatter(pca_random.iloc[:,0], pca_random.iloc[:,1],s=2,c='green', alpha=0.2, label='Random')
    axes[0].legend()
    axes[0].set_xlabel('PCA 1')
    axes[0].set_ylabel('PCA 2')
    axes[0].set_xlim([-0.5,3])
    axes[0].set_ylim([-0.5,0.5])

    axes[1].set_title('TSNE results',
              fontsize=20,
              color='red',
              pad=10)

    axes[1].scatter(tsne_results.iloc[:train_images.shape[0], 0].values, tsne_results.iloc[:train_images.shape[0],1].values,s=2,
            c='black', alpha=0.2, label='Original')
    axes[1].scatter(tsne_results.iloc[train_images.shape[0]:train_images.shape[0]+predictions.shape[0],0], tsne_results.iloc[train_images.shape[0]:train_images.shape[0]+predictions.shape[0],1],s=2,
            c='red', alpha=0.2, label='Synthetic')
    axes[1].scatter(tsne_results.iloc[train_images.shape[0]+predictions.shape[0]:,0], tsne_results.iloc[train_images.shape[0]+predictions.shape[0]:,1],s=2,
            c='green', alpha=0.2, label='Random')
    #axes[1].set_xlim([-0.5,2])
    #axes[1].set_ylim([-1,1])
    axes[1].legend()
    axes[1].set_xlabel('TSNE 1')
    axes[1].set_ylabel('TSNE 2')
    
    fig4.suptitle('Validating synthetic vs real data diversity and distributions',
             fontsize=16,
             color='grey')
    plt.savefig('/home/giacomo/Documents/Synthetic_dataset/'+cd_save+'/PCA_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    
    SAVE=input('Do you want to save the examples? ')
    if SAVE=='Y':
        np.save('/home/giacomo/Documents/Synthetic_dataset/Generated_spectrograms'+'/U_Spect_residuals',np.array(pred_im))

    
train(train_dataset,random_dataset, EPOCHS,train_images)



