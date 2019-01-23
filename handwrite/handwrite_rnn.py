#此為rnn練習

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


tf.set_random_seed(1)   # set random seed


# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限,要循環多少次
batch_size = 128            
n_inputs = 28               # MNIST data input 行 (img shape: 28*28)
n_steps = 28                # time steps列
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    #hidden layer for input to cell
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs) 二維數據
    X = tf.reshape(X, [-1, n_inputs]) #n_input =28
    X_in = tf.matmul(X, weights['in']) + biases['in'] # X_in = W*X + b
    # X_in ==> (128 batches , 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    

    #cell
    # 定義basic LSTM Cell 這邊使用 BasicLSTMCell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 主線state =c_state , 分線state= m_state
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) #初始化全零 state


    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    #hidden layer for output as the final results
    results = tf.matmul(final_state[1], weights['out']) + biases['out'] #把cell的output拿出來再乘以w和b
    
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)) #loss
train_op = tf.train.AdamOptimizer(lr).minimize(cost) #減少loss

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #準確度

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters: #每次循環時
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs]) 
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0: #每20次顯示一次準確度
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1

