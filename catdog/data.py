"""
data.py
~~~~~~~
"""

# Import libraries
import cv2
import os
import random
import numpy as np

from consts import Const


# constants
TRAIN_DIR = '/Users/Nolsigan/Documents/kaggle/data/catdog/train/'
TEST_DIR = '/Users/Nolsigan/Documents/kaggle/data/catdog/test/'


# read single image
def read_image(file_path):
    img = cv2.imread(file_path)
    return cv2.resize(img, (Const.ROWS, Const.COLS), interpolation=cv2.INTER_CUBIC)


# prepare data
def prepare_data(images):
    count = len(images)
    data = np.ndarray((count, Const.CHANNELS, Const.ROWS, Const.COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i % 250 == 0:
            print 'processed {} of {}'.format(i, count)

    return data


# fetch whole train data
def fetch_train_data():
    # images paths
    images_path = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if '.jpg' in i]
    random.shuffle(images_path)

    # load images
    data = prepare_data(images_path)

    # labels
    labels = []
    for image_path in images_path:
        if 'dog.' in image_path:
            labels.append(1)
        else:
            labels.append(0)

    return data, labels


def fetch_test_data():
    # images paths
    images_path = [TEST_DIR+i for i in os.listdir(TEST_DIR) if '.jpg' in i]

    # load images
    data = prepare_data(images_path)

    return data