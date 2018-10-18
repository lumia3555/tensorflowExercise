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
  # 通过with操作，使得scope内生成的variable自动命名为conv1/xxx,便于区分不同卷积层之间的组件
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

  # 最大池化层
  # padding=VALID 表示取样时不能超过边框
  # padding=SAME  表示取样时可以填充边界外的点
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
  # 获取每个样本的一维维度
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

# 时间评估
def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0

  for i in range(num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print('%s: step :%d, duration =%.3f' %(datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
    
  mn = total_duration / num_batches
  vr = total_duration_squared / num_batches - mn * mn
  sd = math.sqrt(vr)
  print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(datetime.now(), info_string, num_batches, mn, sd))

# 定义主函数
def run_benchmark():
  with tf.Graph().as_default():
    image_size = 224
    # 并不使用实际ImageNet数据训练，只使用随机图片数据测试前馈和反馈计算的耗时
    # 使用random_normal创建标准差0.1的正态分布
    images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
    # 为什么是用的pool5而不是fc3
    pool5, parameters = inference(images)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    time_tensorflow_run(sess, pool5, "Forward")
    # 为最后的输出设置一个优化目标
    objective = tf.nn.l2_loss(pool5)
    # 求相对于loss的所有模型参数的梯度
    grad = tf.gradients(objective, parameters)
    time_tensorflow_run(sess, grad, "Forward-backward")

# 执行主函数
run_benchmark()

