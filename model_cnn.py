from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import read_data
import numpy as np
import tensorflow as tf

NUM_TRAIN_EXAMPLES = read_data.NUM_TRAIN_EXAMPLES
NUM_CLASSES = read_data.NUM_CLASSES
DROP_PROB = 0.5
REG_STRENGTH = 0.01
INITIAL_LEARNING_RATE = 1e-3
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 5
MOVING_AVERAGE_DECAY = 0.9999
BATCH_SIZE = 128


# Use tf.get_variable() instead of tf.Variable() to be able to reuse variables for evaluation run
# This was necessary when sharing variables between train and eval run.
# Not necessary now as eval run is based off saved checkpoints, which have moving average of the variables
def _variable_with_weight_decay(name, shape, stddev, wd):
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='reg_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images):
    # conv 1
    with tf.variable_scope('conv1') as scope:
        weights = _variable_with_weight_decay('weights', shape=[5, 5, 3, 32], stddev=1/np.sqrt(5*5*3), wd=0.00)
        biases = tf.get_variable('biases', shape = [32], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    # conv 2
    with tf.variable_scope('conv2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[5, 5, 32, 64], stddev=1/np.sqrt(5*5*32), wd=0.00)
        biases = tf.get_variable('biases', shape = [64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv1, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # pool 1
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # conv 3
    with tf.variable_scope('conv3') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1/np.sqrt(3*3*64), wd=0.00)
        biases = tf.get_variable('biases', shape = [64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)

    # conv 4
    with tf.variable_scope('conv4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1/np.sqrt(3*3*64), wd=0.00)
        biases = tf.get_variable('biases', shape = [64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv3, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)

    # pool 2
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # conv 5
    with tf.variable_scope('conv5') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1/np.sqrt(3*3*64), wd=0.00)
        biases = tf.get_variable('biases', shape = [64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(pool2, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)

    # conv 6
    with tf.variable_scope('conv6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1/np.sqrt(3*3*64), wd=0.00)
        biases = tf.get_variable('biases', shape = [64], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(conv5, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(bias, name=scope.name)

    # pool 3
    with tf.variable_scope('pool3') as scope:
        pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    # fully connected 1
    with tf.variable_scope('fc1') as scope:
        batch_size = images.get_shape()[0].value
        pool3_flat = tf.reshape(pool3, [batch_size, -1])
        dim = pool3_flat.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=1/np.sqrt(dim), wd=REG_STRENGTH)
        biases = tf.get_variable('biases', shape = [384], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(pool3_flat, weights) + biases, name=scope.name)

    # fully connected 2
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=1/np.sqrt(384), wd=REG_STRENGTH)
        biases = tf.get_variable('biases', shape = [192], initializer=tf.constant_initializer(0.0))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

    # dropout
        fc2_drop = tf.nn.dropout(fc2, DROP_PROB)

    # Softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', shape=[192, NUM_CLASSES], stddev=1/np.sqrt(192), wd=0.000)
        biases = tf.get_variable('biases', shape = [NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc2_drop, weights), biases, name=scope.name)

    return logits


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    data_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', data_loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    data_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', data_loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return total_loss


def training(total_loss):

    global_step = tf.Variable(0, name='global_step', trainable=False)
    decay_steps = int(EPOCHS_PER_LR_DECAY * NUM_TRAIN_EXAMPLES / BATCH_SIZE)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LR_DECAY_FACTOR, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    opt_op = optimizer.minimize(total_loss, global_step=global_step)

    mov_average_object = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    moving_average_op = mov_average_object.apply(tf.trainable_variables())

    with tf.control_dependencies([opt_op]):
        train_op = tf.group(moving_average_op)

    return train_op


def evaluation(logits, true_labels):
    correct_pred = tf.nn.in_top_k(logits, true_labels, 1)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))*100