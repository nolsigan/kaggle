"""
data.py
~~~~~~~
"""

# Import libraries
from PIL import Image
import os
import random
import numpy as np

from consts import Const


# read single image
def read_image(file_path):
    img = Image.open(file_path)
    img = img.resize((Const.ROWS, Const.COLS), Image.ANTIALIAS)
    return np.swapaxes(np.array(img), 0, 2)


# prepare data
def prepare_data(images):
    count = len(images)
    data = np.ndarray((count, Const.CHANNELS, Const.ROWS, Const.COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        data[i] = read_image(image_file)
        if i % 250 == 0:
            print 'processed {} of {}'.format(i, count)

    return data


# fetch whole train data
def fetch_train_data():
    # images paths
    images_path = [Const.TRAIN_DIR+i for i in os.listdir(Const.TRAIN_DIR) if '.jpg' in i][:500]
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
    images_path = [Const.TEST_DIR+i for i in os.listdir(Const.TEST_DIR) if '.jpg' in i]

    # load images
    data = prepare_data(images_path)

    return data
