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
from itertools import combinations

conv_activation='relu'

Nlat=128
Nlon=128
n_channels=2

valN=200
input_normalized_transfer=np.zeros([valN,Nlon, Nlat,n_channels],np.float32)

Filename = './FDNS Psi W_Re_100k.mat'

with h5py.File(Filename, 'r') as f:
  input_normalized_transfer[:,:,:,1]=np.array(f['W'],np.float32).T
  input_normalized_transfer[:,:,:,0]=np.array(f['Psi'],np.float32).T
  f.close()

def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([

            ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(Nlon,Nlat,n_channels),trainable = False),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, trainable = False),
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

model = build_model(**params)

combs = list(combinations(range(1,12),2))+list(combinations(range(1,12),1))
#combs = list([(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,)])
print(len(combs))

test_origin = np.reshape(input_normalized_origin[0:200:5,:,:,:],(40,128,128,2))
test_TL = np.reshape(input_normalized_transfer[0:200:5,:,:,:],(40,128,128,2))

for k in [10]:
    for j in combs:
        TL_layers = np.array(j)
        TL_layers = [str(i) for i in TL_layers]
        layers_str = ''

        for layer in TL_layers:
          layers_str += '_'+ layer

        layer_activations = dict([])

        model.load_weights('./weights_TL_from_Re_1k_to_Re_100k_per_train_'+str(k)+'_layers'+layers_str)
        for i in range(11):
            inp = model.inputs

            layer = model.layers[i]
            out = layer.output

            inter_output = keras.backend.function([inp],[out])

            layer_activations['activations_'+str(i+1)] = np.squeeze(inter_output(test_TL))

        savemat('./activations_TL_from_Re_1k_to_Re_100k_per_train_' + str(k) + '_layers' + layers_str + '_40_samples.mat',layer_activations)
