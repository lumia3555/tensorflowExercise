#!/usr/bin/python
# -*- coding:utf8 -*-

# softmax预测实现MNIST手写字识别

# 获取MNIST数据集，该数据集中有55000个训练样本，且图片大小均为28*28=784，
# 所以训练数据的输入是一个55000*784的tensor，输出应该是一个55000*10的tensor
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 载入tensorflow，创建一个新的InteractiveSession
# 这个命令会将当前session设置为默认session，后续的运算也跑在此session中，不同session之间的数据和运算相互独立
import tensorflow as tf
sess = tf.InteractiveSession()

# 设置超参数
learning_rate = 0.5
batch_size = 100
training_epochs = 1000

# 创建placeholder，即输入数据的地方，第一个参数是数据类型，第二个参数为tensor的shape，None表示不限
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 为模型中的weights和bias创建Variable对象。Variable是存储模型参数的，在迭代中持续更新。
W = tf.Variable(tf.zeros([784, 10]))  # weights
b = tf.Variable(tf.zeros([10]))       # bias

# 定义预测函数，使用softmax实现分类，即hypothesis
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义代价函数，即cost function/loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# 定义优化算法，此处使用常见的随机梯度下降SGD，并传入预先设置好的学习率和损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# tensorflow的全局参数初始化器，并执行其run方法
tf.global_variables_initializer().run()

# 迭代执行训练操作
for epoch in range(training_epochs):
  # 每次随机从训练集中获取batch_size条样本组成一个mini-batch
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  # 传入优化器中运行
  optimizer.run({x: batch_xs, y: batch_ys})

# 获取与真实结果比对的tensor，相同则为1，不同则为0
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# 计算平均值，即为准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
