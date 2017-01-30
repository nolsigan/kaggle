# Kaggle

Nolsigan's code for kaggle competitions using Tensorflow & keras!

Only deals with problems that can be solved with deep learning.

## Cat vs Dog

* simple_vgg
    * Uses simplified version of VGG
    * After epoch 8, train loss still decreases but validation loss doesn't (overfitting)
    * __Achieved 85.65% accuracy__

## MNIST

* simple_nn.ipynb
    * Uses simple neural network model with two 128-neurons hidden layers
    * Used extern data from official MNIST site, gaining 33% more data than Kaggle provides.
    * Loss decreases to zero after 100 epochs. ( Maybe this model is enough for this data? )
    * __Achieved 99.714% accuracy__

* conv2d.ipynb
    * Uses CNN model of tensorflow with two convolutional layers and max pooling, a 512-neurons hidden layer
    * Training speed is way slower than simple_nn. Local machine had hard time just training for 10 epochs.
    * Loss is 0.05 after 10 epochs, but still achieves pretty good Result.
    * __Achieved 99.38% accuracy__

## Titanic

* gender.ipynb ( Kaggle Tutorial )
    * Uses simple classification using only gender property.
    * __Achieved 77.65% accuracy__

* simple_nn.ipynb
    * Uses simple neural network model with two 128-neurons hidden layers
    * Result is worse than simple gender model. Not suitable for neural network model.
    * __Achieved 66.96% accuracy__
