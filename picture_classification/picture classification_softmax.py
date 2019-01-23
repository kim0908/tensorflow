'''Softmax-Classifier for CIFAR-10'''
#此為softmax分類器的練習
#圖片來源使用 CIFAR-10,每張圖片32x32,彩圖所以為 32x32x3顏色(RGB)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import data_helper #另一個py有定義此funtion,負責讀取包含數據集的文件，並把數據放入一個方便我們操作的數據結構中

beginTime = time.time() #計時器,用來測量運行的時間

# Parameter definitions
batch_size = 100
learning_rate = 0.005
max_steps = 1000

# Uncommenting this line removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)

# Prepare data
data_sets = data_helper.load_data() #將6萬張圖分為兩個部分,5萬train,1萬test

#load_data()返回
#(50000, 3072) = images_train = train data->50000x3072
#(50000,)      = labels_train = train data set的50000個標籤（每個數字從0到9代表圖像訓練集的10個分類）
#(10000, 3072) = images_test =test data->10000x3072
#(10000,)      = labels_test = test data set的10000個標籤
#classes：10個分類，將0-9每個數字代表一樣（0代表'飛機',1代表'車'...）


# -----------------------------------------------------------------------------
# Prepare the TensorFlow graph
# only defining the graph here
# -----------------------------------------------------------------------------

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072]) #none是因為希望可以隨時改變實際input image的個數
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables (these are the values we want to optimize)
weights = tf.Variable(tf.zeros([3072, 10])) #image是[1x3072],去乘一個[3072x10]的weight矩陣(可能性的高低),得到[1x10]分類矩陣
biases = tf.Variable(tf.zeros([10])) #bias避免起始值全部都0,則不管怎麼乘都是0的狀況

# Define the classifier's result
logits = tf.matmul(images_placeholder, weights) + biases # y=wx+b

# Define the loss function (use softmax_cross_entropy)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
  labels=labels_placeholder))

# Define the training operation 
#也可以用這種但準確率較低->train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())

  # Repeat max_steps times
  for i in range(max_steps):

    # Generate input data batch
    #batch是指從train data set中隨機收取幾張圖和標籤
    #解釋batch,epoch,iteratio-->對於一個有 2000 個訓練樣本的數據集,將2000個樣本分為大小為500的batch,那么完成一个epoch(完整的set通過nn並返回一次)需要4个 iteration
    #batch size(one batch中的圖數量) 可告訴我們參數更新的頻率
    indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
    images_batch = data_sets['images_train'][indices]
    labels_batch = data_sets['labels_train'][indices]

    # Periodically print out the model's current accuracy
    if i % 100 == 0: #每100次,print一次準確率
      train_accuracy = sess.run(accuracy, feed_dict={
        images_placeholder: images_batch, labels_placeholder: labels_batch})
      print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

    # Perform a single training step
    sess.run(train_step, feed_dict={images_placeholder: images_batch,
      labels_placeholder: labels_batch})

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: data_sets['images_test'],
    labels_placeholder: data_sets['labels_test']})
  print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
