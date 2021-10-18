import tensorflow as tf
tf.random.set_seed(10)
import random
random.seed()

import os
import sys
import numpy as np

from keras import layers

from tensorflow.python.keras.backend import clear_session

from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
from keras.layers import Dense
from keras import Sequential
import h5py
import keras

from scipy.io import loadmat,savemat


train_orig = 2000

percent = 10

trainN = int(int(percent)*.01*train_orig)

valN=200
lead=1;
batch_size = 16
num_epochs = 50
pool_size = 2
drop_prob=0.0
conv_activation='relu'
Nlat=128
Nlon=128
n_channels=2
NT = 3000 # Numer of snapshots per file
numDataset = 1 # number of dataset / 2

## Data set has 100k data points


print('Start....')

input_normalized=np.zeros([NT,Nlon, Nlat,n_channels],np.float32)
output_normalized=np.zeros([NT,Nlon,Nlat,1],np.float32)
input_normalized_val=np.zeros([valN,Nlon, Nlat,n_channels],np.float32)
output_normalized_val=np.zeros([valN,Nlon,Nlat,1],np.float32)

# Validation data input
Filename = './FDNS Psi W_val.mat'

with h5py.File(Filename, 'r') as f:
  input_normalized_val[:,:,:,1]=np.array(f['W'],np.float32).T
  input_normalized_val[:,:,:,0]=np.array(f['Psi'],np.float32).T
  f.close()

# Validation data output
Filename = './FDNS PI_val.mat'
with h5py.File(Filename, 'r') as f:
  output_normalized_val[:,:,:,0]=np.array(f['PI'],np.float32).T
  f.close()

Filename = './FDNS Psi W_train.mat'

with h5py.File(Filename, 'r') as f:
  input_normalized[0:NT,:,:,1]=np.array(f['W'],np.float32).T
  input_normalized[0:NT,:,:,0]=np.array(f['Psi'],np.float32).T
  f.close()

# Training data output
Filename = './FDNS PI_train.mat'
with h5py.File(Filename, 'r') as f:
  output_normalized[0:NT,:,:,0]=np.array(f['PI'],np.float32).T
  f.close()

  index=np.random.permutation(trainN)
  input_normalized=input_normalized[index,:,:,:]
  output_normalized=output_normalized[index,:,:,:]

def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([

            ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlon,Nlat,n_channels),trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),

            ] + [keras.layers.Dense(hidden_size, activation='sigmoid', trainable = False) for i in range(n_hidden_layers)] +

            [

            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,trainable = False),
            layers.Convolution2D(1, kernel_size, padding='same', activation=None,trainable = False)
            ]
            )
    optimizer= keras.optimizers.Adam(lr=lr)


    model.compile(loss='mean_squared_error', optimizer = optimizer)

    return model


params = {'conv_depth': 64, 'hidden_size': 5000,
              'kernel_size': 5, 'lr': 0.00005, 'n_hidden_layers': 0}


for j in range(1,12):
    TL_layers = str(j)
    layers_str = '_' + TL_layers


    print('Finish Initialization')
    print(np.shape(input_normalized))
    print('Memory taken by training input:')
    print(input_normalized.nbytes)
    print('Memory taken by training output:')
    print(np.shape(output_normalized))
    print(output_normalized.nbytes)

    print('Memory taken by validation input:')
    print(input_normalized_val.nbytes)
    print('Memory taken by validation output:')
    print(np.shape(output_normalized))
    print(output_normalized_val.nbytes)


    reset_keras()

    model = build_model(**params)

    model.load_weights('./weights_cnn_KT_N_2700_Re_1k_Rotate')


    layer = model.layers[j-1]
    layer.trainable = True

    print(model.summary())
    optimizer= keras.optimizers.Adam(lr=params['lr'])

    model.compile(loss='mean_squared_error', optimizer = optimizer)

    loss = []
    val_loss = []

    hist = model.fit(input_normalized[0:trainN,:,:,:], output_normalized[0:trainN,:,:,:],
                 batch_size = batch_size,shuffle='True',
                 verbose = 0,
                 epochs = num_epochs,
                 validation_data=(input_normalized_val[:,:,:,:],output_normalized_val[:,:,:,:]))

    loss = np.hstack([loss,hist.history['loss']])
    val_loss = np.hstack([val_loss,hist.history['val_loss']])
    print(val_loss)


    savemat('./Loss_TL_from_Re_1k_to_Re_100k_per_train_p_' + percent + '_layers'+layers_str + '.mat',dict([('loss',loss),('val_loss',val_loss)]))
    model.save_weights('./weights_TL_from_Re_1k_to_Re_100k_per_train_p_'+percent+'_layers'+layers_str)
    model.save('./Models/CNN_TL_from_Re_1k_to_Re_100k_per_train_p_'+percent+'_layers'+layers_str)
    prediction=model.predict(input_normalized_val[0:100,:,:,:])


    savemat('./prediction_TL_from_Re_1k_to_Re_100k_per_train_p_'+percent+'_layers'+layers_str+'.mat',dict([('test',output_normalized_val[:100,:,:,:]),('input',input_normalized_val[0:100,:,:,:]),('prediction',prediction[0:100,:,:])]))
