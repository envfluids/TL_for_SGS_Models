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

combs = list(combinations(range(1,10),1))
print(len(combs))

for k in [10]:
    for j in combs:
        TL_layers = np.array(j)
        TL_layers = [str(i) for i in TL_layers]
        layers_str = ''

        for layer in TL_layers:
          layers_str += '_'+ layer

        layer_weights = dict([])
        layer_bias = dict([])

        model.load_weights('./weights_TL_from_Re_1k_to_Re_100k_per_train_'+str(k)+'_layers'+layers_str)

        for i in range(11):
          layer = model.layers[i]
          weights = layer.get_weights()

          layer_weights['l'+str(i+1)+'_w'] = weights[0]
          layer_bias['l'+str(i+1)+'_b'] = weights[1]

        savemat('./Weights_Matlab/Weights_TL_from_Re_1k_to_Re_100k_per_train_' + str(k) + '_layers' + layers_str + '.mat',layer_weights)
        savemat('./Weights_Matlab/Biases_TL_from_Re_1k_to_Re_100k_per_train_' + str(k) + '_layers' + layers_str + '.mat',layer_bias)
