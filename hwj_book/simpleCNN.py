#!/usr/bin/python
# -*- coding:utf8 -*-

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf 
sess = tf.InteractiveSession()

learning_rate = 1e-4
training_epochs = 5000
batch_size = 50

# 初始化weights的函数，设置为固定shape的标准差为0.1的正态分布
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# 初始化bias的函数，设置为固定shape的常数0.1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# tf.nn.conv2d(input, filter, strides, padding)的参数设定
# input - [样本容量(-1表示未知), height, width, channel]
# filter - [height, width, input_channel, output_channel]
# strides - [1,f,f,1]
# padding - 'SAME' / 'VALID'
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_22(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# gray image has 1 channel, RGB has 3
# 这里不能设定成784的一维向量了，要转成图片形式的28*28
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 构建网络模型
# 卷积层1，输入图片尺寸28*28
W_conv1 = weight_variable([5,5,1,32]) # 初始化weights，即卷积核，大小5*5，入通道数1，出通道数32
b_conv1 = bias_variable([32]) # 初始化bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_22(h_conv1)

# 卷积层2，输入图片尺寸14*14
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_22(h_conv2)

# 全连接 输入图片尺寸为7*7,输出通道数64，转变为1维（7*7*64），并映射成1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout - 全连接完了再dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 从全连接层到输出分类层的映射1024 -> 10，直接用softmax判断
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))

# 优化器选用Adam
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# predict
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
tf.global_variables_initializer().run()
for epoch in range(training_epochs):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  if epoch % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(epoch, train_accuracy))
  optimizer.run(feed_dict={x:batch_xs, y:batch_ys, keep_prob: 0.5})


print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))