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

    model.add(ZeroPadding2D((2, 2), input_shape=(2, 8, 24)))
    # input shape = (2, 12, 28)
    model.add(Conv2D(128, (5, 5), activation='relu', kernel_initializer=init))
    # input shape = (128, 8, 24)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # input shape = (128, 4, 12)
    model.add(ZeroPadding2D((1, 0)))
    # input shape = (128, 6, 12)
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=init))
    # input shape = (128, 4, 10)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # input shape = (128, 2, 5)
    model.add(Conv2D(bottleneck_width, (2, 5), activation='relu',
        kernel_initializer=init))

    # middle of network. input shape = (bottleneck_width, 1, 1)
    model.add(Conv2DTranspose(128, (2, 4), strides=(2, 2), activation='relu',
        kernel_initializer=init, data_format=keras.backend.image_data_format()))
    # input shape = (128, 2, 4)
    model.add(Conv2DTranspose(128, (2, 5), strides=(2, 2), activation='relu',
        kernel_initializer=init, data_format=keras.backend.image_data_format()))
    # input shape = (128, 4, 11)
    model.add(Conv2DTranspose(2, (2, 4), strides=(2, 2), activation='relu',
        kernel_initializer=init, data_format=keras.backend.image_data_format()))
    #output shape = (2, 8, 24)
    return model

def compile_model(model):
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9),
        metrics=['mse'])
