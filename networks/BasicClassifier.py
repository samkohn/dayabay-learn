"""
Build a basic convolutional classifier using the Keras framework.

"""

import numpy as np
import h5py
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.initializers import TruncatedNormal

def get_models(bottleneck_width):
    """
    Construct a (classifier, encoder) model and return the two
    models in a tuple in that order.

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
    conv3a = Flatten()(conv3)
    dense1 = Dense(128, activation='relu')(conv3a)
    dense1a = Dropout(0.3, seed=1739)(dense1)
    dense2 = Dense(16, activation='relu')(dense1a)
    dense2a = Dropout(0.3, seed=1740)(dense2)
    dense3 = Dense(2, activation='softmax')(dense2a)

    encoder = Model(input_layer, conv3, name='encoder')
    classifier = Model(input_layer, dense3, name='classifier')

    return (classifier, encoder)

def compile_model(model):
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),
        metrics=['categorical_accuracy'])

