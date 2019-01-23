#此為簡單cnn練習

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST手寫數字圖片庫,書入數據
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定義準確度
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_prob:1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys,keep_prob:1.0})
    return result

def weight_variable(shape):  #輸入一個shap返回variable定義參數
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):  #定義一開始是0.1
	initial=tf.constant(0.1,shape=shape) 
	return tf.Variable(initial)

#定義一個二維的covolution
def conv2d(x,W):  #x input圖片所有訊息 w是wight
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
        #strides[1,x_movement,y_movement,1]
        #padding選用same padding的方式

def max_pool_2x2(x): 
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #stride movement設2所以移動較快,圖片被壓縮,ksize=kernel size=patch size

#定義holder
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32) #dropout 避免overfitting

#因為xs是包括所有sample所以要改成
x_image=tf.reshape(xs,[-1,28,28,1]) #最後的1是channel 黑白照片是1彩色是3

#定義conv1 layer
W_conv1=weight_variable([5,5,1,32]) #定義weight/ patch 5x5 ,in size=1 輸入的圖片厚度為1,out size=32
b_conv1=bias_variable([32]) #定義bias
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #激勵函數,output size=28x28x32
h_pool1=max_pool_2x2(h_conv1) #pooling處理,output size=14x14x32

#定義conv2 layer
W_conv2=weight_variable([5,5,32,64]) #定義weight/ patch 5x5 ,in size=32 厚度為32,out size=64
b_conv2=bias_variable([64]) #定義bias
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #激勵函數,output size=14x14x64
h_pool2=max_pool_2x2(h_conv2) #pooling處理,output size=7x7x64

#定義fully connected layer
#func1 layer
W_fc1=weight_variable([7*7*64,1024]) #(in size,out size)
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#想把[n_sample,7,7,64]-->[n_sample,7*7*64]
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1) #[n_sample,7*7*64]*weight+bias
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob) #dropout

#func2 layer
W_fc2=weight_variable([1024,10]) #(in size,out size)
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2) #classification分類器



# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))

#gradient decent
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#開始訓練，每次取100張圖片，每訓練50次輸出一下預測精準
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100) #每次取100張圖片
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys , keep_prob:1.0})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[0:2000], mnist.test.labels[0:2000]))
