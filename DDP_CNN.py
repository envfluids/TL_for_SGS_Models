import tensorflow as tf
tf.random.set_seed(10)
import random
random.seed()

import os
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


trainN=2000
valN=200
lead=1;
batch_size = 16
num_epochs = 100
pool_size = 2
drop_prob=0.0
conv_activation='relu'
Nlat=128
Nlon=128
n_channels=2
NT = 2000 # Numer of snapshots per file
numDataset = 1 # number of dataset / 2

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
Filename = './FDNS Psi W_train.mat'
with h5py.File(Filename, 'r') as f:
  output_normalized[0:NT,:,:,0]=np.array(f['PI'],np.float32).T
  f.close()

index=np.random.permutation(trainN)
input_normalized=input_normalized[index,:,:,:]
output_normalized=output_normalized[index,:,:,:]

def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([

            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlon,Nlat,n_channels)),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            ] + [keras.layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +

            [

            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.Convolution2D(1, kernel_size, padding='same', activation=None)
            ]
            )
    optimizer= keras.optimizers.Adam(lr=lr)


    model.compile(loss='mean_squared_error', optimizer = optimizer)

    return model


params = {'conv_depth': 64, 'hidden_size': 5000,
              'kernel_size': 5, 'lr': 0.00005, 'n_hidden_layers': 0}

model = build_model(**params)

print(model.summary())

hist = model.fit(input_normalized[0:trainN,:,:,:], output_normalized[0:trainN,:,:,:],
             batch_size = batch_size,shuffle='True',
             verbose=1,
             epochs = num_epochs,
             validation_data=(input_normalized_val[:,:,:,:],output_normalized_val[:,:,:,:]))

model.save('./CNN_N_2000')
model.save_weights('./weights_CNN_N_2000')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
savemat('loss_N_2000_Re_1k.mat' ,dict([('trainLoss',loss),('valLoss',val_loss)]))


prediction=model.predict(input_normalized_val[0:100,:,:,:])

savemat('prediction_KT_N_1900_Re_1k.mat',dict([('test',output_normalized_val[:100,:,:,:]),('input',input_normalized_val[0:100,:,:,:]),('prediction',prediction[0:100,:,:])]))
