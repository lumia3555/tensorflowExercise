#!/usr/bin/python
# -*- coding:utf8 -*-

# softmax实现识别MNIST手写字基础上，添加MLP
# 与softmax.py的区别：
#   1. 不是输入到输出的直接映射，而是中间多了一层hidden layer
#   2. 在隐藏层中使用了relu激活函数和随机失活dropout
#   3. 优化器使用了可以动态改变学习率的Adagrad而非普通的SGD

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()

# hyper parameters
learning_rate = 0.3
batch_size = 100
training_epochs = 3000

in_units = 784  # 输入层维度
h1_units = 300  # 隐藏层1维度

# 手动设置从输入层到隐藏层1的参数，包括weights和bias
# weights设置时初始化权重为[784, 300]的标准差为0.1的正态分布
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
# bias设置为0
b1 = tf.Variable(tf.zeros([h1_units]))

# 从隐藏层1到输出层的映射，weights和bias均为0
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
# 设置丢失率,dropout rate
keep_prob = tf.placeholder(tf.float32)

# 生成隐藏层1的方法，对x*W1 + b1的结果采用relu激活函数
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# 根据设置的丢失率进行dropout
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

pred = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# ground truth
y = tf.placeholder(tf.float32, [None, 10])

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# use Adagrad instead of SGD
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

tf.global_variables_initializer().run()

# train
for epoch in range(training_epochs):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  # 训练时只保留75%的节点
  optimizer.run({x: batch_xs, y: batch_ys, keep_prob: 0.75})

# predict
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 预测时保留100%节点
print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
