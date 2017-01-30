"""
models.py
~~~~~~~~~
"""
# Import libraries
from keras.models import Sequential
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import RMSprop

from consts import Const

import os


def simple_vgg():
    model = Sequential()

    # conv layer 1
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, Const.ROWS, Const.COLS)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # conv layer 2
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # conv layer 3
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # conv layer 4
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # dense layer
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # load weights if exists & compile
    if os.path.isfile(Const.WEIGHT_CHECKPOINT):
        model.load_weights(Const.WEIGHT_CHECKPOINT)

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

    return model
