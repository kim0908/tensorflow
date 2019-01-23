#此為簡單手寫辨識練習 用mnist

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定義加入一層神經網路
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#定義準確度
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


#holder每張圖片像素28x28 所以input應該是28x28=784
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10]) #y有10個輸出，數字0-9

#定義output #此時此神經網路只有輸入層和輸出層
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))

#gradient decent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#開始訓練，每次取100張圖片，每訓練50次輸出一下預測精準
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100) #每次取100張圖片
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
