"""
conv2d.py
~~~~~~~~~

cat vs dog classification using CNN
"""

# Import libraries
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, BatchNormalization, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
import os
import random
import numpy as np
import cv2


# Constants
TRAIN_DIR = '/Users/Nolsigan/Documents/kaggle/data/catdog/train/'

ROWS = 64
COLS = 64
CHANNELS = 3


# Prepare data
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
train_dogs = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_images = train_dogs[:1000] + train_cats[:1000]
random.shuffle(train_images)


# read image function
def read_image(file_path):
    img = cv2.imread(file_path)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 250 == 0:
            print 'processed {} of {}'.format(i, count)

    return data

train = prep_data(train_images)
print("Train shape : {}".format(train.shape))

labels = []
for i in train_images:
    if 'dog.' in i:
        labels.append(1)
    else:
        labels.append(0)


# Design model
model = Sequential()

model.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(128, 3, 3, border_mode='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

from keras.callbacks import EarlyStopping, Callback

nb_epoch = 10
batch_size = 16


# callback for loss logging every epoch
class LossHistory(Callback):
    losses = []
    val_losses = []

    def on_training_begin(self, logs={}):
        print 'begin training!'
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        print logs.get('loss')
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')



def run_catdog():
    history = LossHistory()
    model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.25, shuffle=True, callbacks=[history, early_stopping])

    # predictions = model.predict(test, verbose=0)
    return history


print "before run!"
history = run_catdog()