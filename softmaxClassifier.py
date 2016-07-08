import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import scipy.io
from scipy.io import savemat, loadmat
from os import listdir, walk
from os.path import isfile, join
#import PIL
import matplotlib.pyplot as plt
import time

start_time = time.time()

trainPath = 'data/train/'
avgImgHeight = int(360/4)       # Actual average height is 360. I am doing more downsampling
avgImgWidth = int(400/4)        # Actual average width is 400. I am doing more downsampling
imgClasses = ['dog', 'cat']
numClasses = len(imgClasses)

fileNames = next(walk(trainPath))[2]
numTrain = len(fileNames)
trainDataX = np.zeros((numTrain, avgImgHeight, avgImgWidth, 3), dtype=np.uint8)
trainDataY = np.zeros((numTrain, numClasses), dtype=np.uint8)
for i, fileName in enumerate(fileNames):
    img = imread(join(trainPath, fileName))
    img = imresize(img, (avgImgHeight, avgImgWidth))
    trainDataX[i, :, :, :] = img

    for j, imgClass in enumerate(imgClasses):
        trainDataY[i, j] = (fileName[0:3] == imgClass)

trainDataX = np.reshape(trainDataX, (trainDataX.shape[0], -1))
print('Time taken = ', time.time()-start_time)

flatImgDim = trainDataX.shape[1]

learningRate = 0.00000001
numIter = 10000
batchSize = 400
X = tf.placeholder(tf.float32, [None, flatImgDim])
y = tf.placeholder(tf.float32, [None, numClasses])

W = tf.Variable(np.float32(np.random.randn(flatImgDim, numClasses)*0.0001))
b = tf.Variable(tf.zeros(numClasses))

y_pred = tf.nn.softmax(tf.matmul(X,W) + b)

crossEntropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(crossEntropy)
correctPrediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(numIter):
    sampleIndices = np.random.choice(np.arange(numTrain), batchSize)
    batch_xs = trainDataX[sampleIndices]
    batch_ys = trainDataY[sampleIndices]
    sess.run(trainStep, feed_dict={X: batch_xs, y: batch_ys})
    if i%500 == 0:
       # print(batch_xs, batch_ys)
        print('Cross Entropy =', sess.run(crossEntropy, feed_dict={X: batch_xs, y: batch_ys}),
              'Training accuracy = ', sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys}))

print('Final Training accuracy = ', sess.run(accuracy, feed_dict={X: trainDataX, y: trainDataY}))

