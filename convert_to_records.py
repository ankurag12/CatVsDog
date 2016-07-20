'''''''''
NEED TO RUN THIS ONLY ONCE!
'''''''''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from os.path import join
import tensorflow as tf
import manage_images

IMG_CLASSES = ['dog', 'cat']

DATA_DIR = 'data/'
TRAIN_DATA_PATH = 'data/train/'
TEST_DATA_PATH = 'data/test/'

IMG_HEIGHT = int(128)
IMG_WIDTH = int(128)
IMG_CHANNELS = 3
NUM_FILES_DATASET = 22500
VALIDATION_SET_FRACTION = 0.1
NUM_TRAIN_EXAMPLES = int((1 - VALIDATION_SET_FRACTION) * NUM_FILES_DATASET)
NUM_VALIDATION_EXAMPLES = int((VALIDATION_SET_FRACTION) * NUM_FILES_DATASET)
NUM_TEST_EXAMPLES = 2500


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = join(DATA_DIR, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),   # NOT assuming one-hot format of original data
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(argv):
    train_images, train_labels = manage_images.read_images(TRAIN_DATA_PATH, IMG_CLASSES,
                                                           IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    test_images, test_labels = manage_images.read_images(TEST_DATA_PATH, IMG_CLASSES,
                                                         IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    # Generate a validation set.
    validation_size = int(VALIDATION_SET_FRACTION * train_images.shape[0])
    validation_images = train_images[:validation_size, :, :, :]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:, :, :, :]
    train_labels = train_labels[validation_size:]

    # Convert to Examples and write the result to TFRecords.
    convert_to(train_images, train_labels, 'train')
    convert_to(validation_images, validation_labels, 'validation')
    convert_to(test_images, test_labels, 'test')


if __name__ == '__main__':
    tf.app.run()


