"""
train.py
~~~~~~~~

train module
- takes data from data.py
- takes model from other files
- trains and saves checkpoints
    - checkpoints should include loss function name
"""
# Import libraries
from data import fetch_train_data
from models import conv_simple
from consts import Const

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


# constants
BATCH_SIZE = 16
EPOCHS = 20
VALIDATION_SPLIT = 0.25


# loss history class
class LossHistory(Callback):
    losses = []
    val_losses = []

    def on_training_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


# training function
def run_train():
    """
    run training and save checkpoints
    :return: None
    """
    # fetch data
    train, labels = fetch_train_data()

    # make model
    model = conv_simple()

    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    history = LossHistory()
    checkpoint = ModelCheckpoint(Const.WEIGHT_CHECKPOINT, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # run training
    model.fit(train, labels, batch_size=BATCH_SIZE, nb_epoch=EPOCHS,
              validation_split=VALIDATION_SPLIT, shuffle=True, callbacks=[history, early_stopping, checkpoint])


# actually run training!
run_train()
