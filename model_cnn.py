from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import read_data
import tensorflow as tf


NUM_CLASSES = read_data.NUM_CLASSES
NUM_EPOCHS = 2
BATCH_SIZE = 100
DROP_PROB = 0.5


def inference(images):
    # conv 1
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.001))
        biases = tf.Variable(tf.constant(0.01, shape = [32]))
        conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

        # pool 1
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # norm 1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # conv 2
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.001))
        biases = tf.Variable(tf.constant(0.01, shape = [64]))
        conv = tf.nn.conv2d(norm1, weights, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

        # pool 2
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # norm 2
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')

    # fully connected 1
    with tf.variable_scope('fc1') as scope:
        norm2_flat = tf.reshape(norm2, [BATCH_SIZE, -1])
        dim = norm2_flat.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal([dim, 384], stddev=0.001))
        biases = tf.Variable(tf.constant(0.01, shape = [384]))
        fc1 = tf.nn.relu(tf.matmul(norm2_flat, weights) + biases, name=scope.name)

    # fully connected 2
    with tf.variable_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([384, 192], stddev=0.001))
        biases = tf.Variable(tf.constant(0.01, shape = [192]))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

    # dropout
        fc2_drop = tf.nn.dropout(fc2, DROP_PROB)

    # Softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal([192, NUM_CLASSES], stddev=0.001))
        biases = tf.Variable(tf.constant(0.01, shape = [NUM_CLASSES]))
        softmax_linear = tf.nn.softmax(tf.matmul(fc2_drop, weights) + biases, name=scope.name)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    data_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return data_loss


def training(total_loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)

    gloabl_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(total_loss, global_step=gloabl_step)

    return train_op


def evaluation(logits, true_labels):
    correct_pred = tf.nn.in_top_k(logits, true_labels, 1)
    return tf.reduce_sum(tf.cast(correct_pred, tf.int32))