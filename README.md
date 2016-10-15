# Kaggle

Nolsigan's code for kaggle completition using Tensorflow!

Only deals with problems that can be solved with deep learning.


## Titanic

* gender.ipynb ( Kaggle Tutorial )
    * Uses simple classification using only gender property.

#### Achieved 77.65% accuracy

* simple_nn.ipynb
    * Uses simple neural network model with two 128-neurons hidden layers
    * Result is worse than simple gender model. Not suitable for neural network model.

#### Achieved 66.96% accuracy


## MNIST

* simple_nn.ipynb
    * Uses simple neural network model with two 128-neurons hidden layers
    * Used extern data from official MNIST site, gaining 33% more data than Kaggle provides.
    * Loss decreases to zero after 100 epochs. ( Maybe this model is enough for this data? )

#### Achieved 99.714% accuracy

* conv2d.ipynb
    * Uses CNN model of tensorflow with two convolutional layers and max pooling, a 512-neurons hidden layer
    * Training speed is way slower than simple_nn. Local machine had hard time just training for 10 epochs.
    * Loss is 0.05 after 10 epochs, but still achieves pretty good Result.

#### Achieved 99.38% accuracy



