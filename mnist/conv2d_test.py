import tensorflow as tf
import csv
import numpy as np
import os

# Define constants
NUM_CLASSES = 10
INPUT_SIZE = 784

BATCH_SIZE = 100
EVAL_BATCH_SIZE = 1

HIDDEN1_UNITS = 512
# HIDDEN2_UNITS = 128

MAX_EPOCHS = 20

CONV1_STRIDES = [1, 1, 1, 1]
CONV2_STRIDES = [1, 1, 1, 1]

TRAIN_DIR = "./checkpoints"


with open('../data/mnist/test_manipulated.csv', newline='') as csvfile:
    csv_file_object = csv.reader(csvfile, dialect='excel')
    haeder = csv_file_object.__next__()
    
    test=[]
    for row in csv_file_object:
        test.append(row)
    test = np.array(test)
    test = test.astype(np.float)

print(test.shape)

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(
        os.path.join(TRAIN_DIR, 'checkpoint-19.meta'))
    saver.restore(
        sess, os.path.join(TRAIN_DIR, 'checkpoint-19'))
    
    # Retrieve Ops
    logits = tf.get_collection('logits')[0]
    inputs_placeholder = tf.get_collection('inputs')[0]
    labels_placeholder = tf.get_collection('labels')[0]
    
    eval_op = tf.nn.top_k(logits)
    
    # Run evaluation
    labels_feed = test[0::, 0]
    inputs_feed = test.reshape((-1, 28, 28, 1))
    labels_feed = np.array(labels_feed)
    # imgplot = plt.imshow(np.reshape(inputs_feed[0], (28, 28)))
    
    prediction = sess.run(eval_op,
                          feed_dict={inputs_placeholder: inputs_feed,
                                    labels_placeholder: labels_feed})
    print('prediction : ', prediction.indices[0][0])

with open('conv2d.csv', 'w', newline='') as csvfile:
    csv_file_object = csv.writer(csvfile, dialect='excel')
    csv_file_object.writerow(['ImageId', 'Label'])
    for i in range(len(test)):
        csv_file_object.writerow([i+1, prediction.indices[i][0]])
