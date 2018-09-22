#!/usr/bin/python
# -*- coding:utf8 -*-

import cifar10, cifar10_input
import tensorflow as tf 
import numpy as np
import time 

# hyper parameters
max_steps = 3000
hatch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape, stddev, wl):
  var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
  if wl is not None:
    weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
    tf.add_to_collection('losses', weight_loss)
  return var

cifar10.maybe_download_and_extract()

# generate training set
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)

# generate test set
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

# placeholder for inputs

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.float32, [batch_size])

# ------ Phase 1: construct network model, give inference -----------
# layer 1 - convolution

weight_1 = variable_with_weight_loss(shape=[5,5,3,64], stddev=5e-2, wl=0.0)
kernel_1 = tf.nn.conv2d(image_holder, weight_1, [1,1,1,1], padding='SAME')
bias_1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
pool_1 = tf.nn.max_pool(conv_1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# layer 2 - convolution

weight_2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=5e-2, wl=0.0)
kernel_2 = tf.nn.conv2d(norm_1, weight_2, [1,1,1,1], padding='SAME')
bias_2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool_2 = tf.nn.max_pool(norm_2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

# layer 3 - full connect

reshape = tf.reshape(pool_2, [batch_size, -1])
dim = reshape.get_shape()[1].value
print '====='
print dim
weight_3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.004)
bias_3 = tf.Variable(tf.constant(0.1, shape=[384]))
local_3 = tf.nn.relu(tf.matmul(reshape, weight_3) + bias_3)

# layer 4 - full connect

weight_4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias_4 = tf.Variable(tf.constant(0.1, shape=[192]))
local_4 = tf.nn.relu(tf.matmul(local_3, weight_4) + bias_4)

# layer 5 - give inference

weight_5 = variable_with_weight_loss(shape[192, 10], stddev=1/192.0, wl=0.0)
bias_5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local_4, weight_5), bias_5)


# ------ Phase 2: define loss -----------
def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels, name='cross_entropy_per_example'
  )
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss = loss(logits, label_holder)

# ------ Phase 3: define optimizer -----------
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)


# ------ Phase 4: train -----------
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

for step in range(max_steps):
  start_time = time.time()
  image_batch, label_batch = sess.run([images_train, labels_train])
  _, loss_value = sess.run([optimizer, loss], feed_dict={image_holder: image_batch, label_holder: label_batch})
  duration = time.time() - start_time

  if step % 10 == 0:
    examples_per_sec = batch_size / duration
    sec_per_batch = float(duration)

    format_str = ('step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)')
    print (format_str % (step, loss_value, examples_per_sec, sec_per_batch))


# ------ Phase 5: run on test data -----------
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
  image_batch, label_batch = sess.run([images_test, labels_test])
  predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch, label_holder: label_batch})
  true_count += np.sum(predictions)
  step += 1

precision = true_count / total_sample_count
print ('precision @ 1 = %.4f' % precision)
