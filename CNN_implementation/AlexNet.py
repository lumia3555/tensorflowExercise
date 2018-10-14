#!/usr/bin/python
# -*- coding:utf8 -*-

from datetime import datetime
import math
import time
import tensorflow as tf

# hyper parameters
batch_size = 32
num_batches = 100

# helper function, show input tensor's name and shape
def print_activations(t):
  print(t.op.name, ' -> ', t.get_shape().as_list())


def inference(images):
  parameters = []

  # 卷积层一
  with tf.name_scope('conv1') as scope:
    # 设置卷积核
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')

    # 执行卷积操作
    conv = tf.nn.conv2d(images, kernel, [1,4,4,1], padding = 'SAME')

    # 设置偏差值
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')

    # 执行偏差值添加操作
    bias = tf.nn.bias_add(conv, biases)

    # ReLU激活函数
    conv1 = tf.nn.relu(bias, name=scope)

    # 打印结构
    print_activations(conv1)

    # 添加可训练的参数到parameters中
    parameters += [kernel, biases]

  # LRN层
  lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')

  # 最大池化层, VALID表示取样时不能超过边框
  pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')

  print_activations(pool1)

  # 卷积层二, 卷积核变为5*5， 输入64， 输出192
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5,5,64,192], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv2)
  lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
  pool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
  print_activations(pool2)

  # 卷积层三, 卷积核变为3*3， 输入192， 输出384
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3,3,192,384], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

  # 卷积层四, 输入384， 输出256
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3,3,384,256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

  # 卷积层五
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3,3,256,256], dtype=tf.float32, stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1,1,1,1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)
  pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')
  print_activations(pool5)

  # 全连接reshape
  reshape = tf.reshape(pool5, [batch_size, -1])
  dim = reshape.get_shape()[1].value

  # 全连接一
  with tf.name_scope('fc1') as scope:
    weights = tf.Variable(tf.truncated_normal([dim, 4096], dtype=tf.float32, stddev=0.04), name='weights')
    bias = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32, trainable=True, name='bias'))
    fc1 = tf.nn.relu(tf.matmul(reshape, weights) + bias)
    parameters += [weights, bias]
  
  # 全连接二
  with tf.name_scope('fc2') as scope:
    weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=0.04), name='weights')
    bias = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32, trainable=True, name='bias'))
    fc2 = tf.nn.relu(tf.matmul(fc1, weights) + bias)
    parameters += [weights, bias]

  # 全连接三
  with tf.name_scope('fc3') as scope:
    weights = tf.Variable(tf.truncated_normal([4096, 1024], dtype=tf.float32, stddev=0.04), name='weights')
    bias = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32, trainable=True, name='bias'))
    fc3 = tf.nn.relu(tf.matmul(fc2, weights) + bias)
    parameters += [weights, bias]

  return fc3, parameters

