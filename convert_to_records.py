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

img_classes = ['dog', 'cat']

data_dir = 'data/'
train_data_path = 'data/train/'
test_data_path = 'data/test/'

avg_img_height = int(128)  # Actual average height is 360. I am doing more downsampling
avg_img_width = int(128)  # Actual average width is 400. I am doing more downsampling
img_channels = 3
validation_perc = 0.1


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

    filename = join(data_dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(np.argmax(labels[index]))),   # Assuming one-hot format of original data
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(argv):
    train_images, train_labels = manage_images.read_images(train_data_path, img_classes, avg_img_height, avg_img_width)
    test_images, test_labels = manage_images.read_images(test_data_path, img_classes, avg_img_height, avg_img_width)

    # Generate a validation set.
    validation_size = int(validation_perc * train_images.shape[0])
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


