#此為可視化練習

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt#一個可視化模組

#定義添加一層神經網路
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#導入數據x y 再加上noise
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

#利用holder 來定義神經網路輸入
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#layer1 輸入層
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 輸出層
prediction = add_layer(l1, 10, 1, activation_function=None)

#計算預測與真實值得誤差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

#用gradient descent去BP
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
#會話激活
sess = tf.Session()
sess.run(init)

# plot the real data 將圖畫出以達到可視化結果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()#連續的plot不會暫停
plt.show()


for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0]) #先除去lines的線段 因為沒有除去會重複被疊加上去
        except Exception: #避免第一次run這個迴圈時還沒定義ax.plot會被偵錯
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5) #r是紅線 lw是線寬度
        plt.pause(0.1)
