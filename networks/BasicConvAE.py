"""
Build a basic convolutional autoencoder using the Keras framework.

"""

import numpy as np
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2DTranspose

def get_model(bottleneck_width):
    model = Sequential()
    num_features = 8 * 24 * 2
    init = keras.initializers.RandomNormal(mean=0,stddev=1.0/num_features)

    model.add(ZeroPadding2D((2, 2), input_shape=(8, 24, 2)))
    # input shape = (12, 28, 2)
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer=init))
    # input shape = (8, 24, 128)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # input shape = (4, 12, 128)
    model.add(ZeroPadding2D((1, 0)))
    # input shape = (6, 12, 128)
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=init))
    # input shape = (4, 10, 128)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # input shape = (2, 5, 128)
    model.add(Conv2D(bottleneck_width, (2, 5), activation='relu',
        kernel_initializer=init))

    # middle of network. input shape = (1, 1, bottleneck_width)
    model.add(Conv2DTranspose(128, (2, 4), strides=(2, 2), activation='relu',
        kernel_initializer=init))
    # input shape = (2, 4, 128)
    model.add(Conv2DTranspose(128, (2, 5), strides=(2, 2), activation='relu',
        kernel_initializer=init))
    # input shape = (4, 11, 128)
    model.add(Conv2DTranspose(2, (2, 4), strides=(2, 2), activation='relu',
        kernel_initializer=init))
    #output shape = (8, 24, 2)
    return model

def compile_model(model):
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9),
        metrics=['mse'])
