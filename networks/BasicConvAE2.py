"""
Build a basic convolutional autoencoder using the Keras framework.

"""

import numpy as np
import h5py
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Cropping2D
from keras.layers import Conv2DTranspose
from keras.initializers import TruncatedNormal

def get_models(bottleneck_width):
    """
    Construct an (autoencoder, encoder, decoder) model and return the three
    segments in a tuple in that order.

    """
    num_features = 8 * 24 * 2
    init = TruncatedNormal(mean=0,stddev=1.0/np.sqrt(num_features))

    input_layer = Input(shape=(2, 8, 24))
    corruption = Dropout(0.3)(input_layer)
    conv1 = ZeroPadding2D((2, 2))(corruption)
    # input shape = (2, 12, 28)
    conv1 = Conv2D(128, (5, 5), activation='relu',
            kernel_initializer=init)(conv1)
    # input shape = (128, 8, 24)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # input shape = (128, 4, 12)
    conv2 = ZeroPadding2D((1, 0))(pool1)
    # input shape = (128, 6, 12)
    conv2 = Conv2D(128, (3, 3), activation='relu',
            kernel_initializer=init)(conv2)
    # input shape = (128, 4, 10)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # input shape = (128, 2, 5)
    conv3 = Conv2D(bottleneck_width, (2, 5), activation='relu',
            kernel_initializer=init)(pool2)

    # middle of network. input shape = (bottleneck_width, 1, 1)
    deconv1 = Conv2DTranspose(128, (2, 5), activation='relu',
            kernel_initializer=init,
            data_format=keras.backend.image_data_format())(conv3)
    # input shape = (128, 2, 5)
    unpool1 = UpSampling2D(size=(2, 2),
            data_format=keras.backend.image_data_format())(deconv1)
    # input shape = (128, 4, 10)
    deconv2 = Conv2DTranspose(128, (3, 3), activation='relu',
            kernel_initializer=init,
            data_format=keras.backend.image_data_format())(unpool1)
    # input shape = (128, 6, 12)
    crop1 = Cropping2D(cropping=(1, 0),
            data_format=keras.backend.image_data_format())(deconv2)
    # input shape = (128, 4, 12)
    unpool2 = UpSampling2D(size=(2, 2),
            data_format=keras.backend.image_data_format())(crop1)
    # input shape = (128, 8, 24)
    deconv3 = Conv2DTranspose(2, (5, 5), activation='tanh',
            kernel_initializer=init,
            data_format=keras.backend.image_data_format())(unpool2)
    # input shape = (128, 12, 28)
    crop2 = Cropping2D(cropping=(2, 2),
            data_format=keras.backend.image_data_format())(deconv3)
    #output shape = (2, 8, 24)

    encoder = Model(input_layer, conv3, name='encoder')
    autoencoder = Model(input_layer, crop2, name='autoencoder')

    return (autoencoder, encoder)

def compile_model(model):
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9))
