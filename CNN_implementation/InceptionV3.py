#!/usr/bin/python
# -*- coding:utf8 -*-

'''
  Inception V3结构
    前置非Inception模块组的卷积组
    Inception Module模块组
    池化
    线性
    softmax分类输出
'''

import tensorflow as tf 
slim = tf.contrib.slim
trunc_normal = lambda stddev:tf.truncated_normal_initializer(0.0, stddev)

# 生成网络中经常使用到的函数的默认参数
def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
  # batch normalization参数字典
  batch_norm_params = {
    'decay': 0.9997,  # 衰减系数
    'epsilon': 0.001,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
    'variables_collections': {
      'beta': None,
      'gamma': None,
      'moving_mean': [batch_norm_var_collection],
      'moving_variance': [batch_norm_var_collection],
    }
  }

  # 给函数参数赋予某些默认值
  with slim.arg_scope(
    # 对slim.conv2d和slim.fully_connected两个函数中的weights_regularizer参数进行默认赋值
    [slim.conv2d, slim.fully_connected],
    weights_regularizer = slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
      # 对slim.conv2d函数的各种参数进行赋值
      [slim.conv2d],
      weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
      activation_fn = tf.nn.relu,
      normalizer_fn = slim.batch_norm,
      normalizer_params = batch_norm_params) as sc:
      return sc

# 定义网络结构
def inception_v3_base(inputs, scope=None):
  
  # 保存某些关键节点以供后续使用
  end_points = {}

  '''
  网络结构综述：
    1. 首先是5个卷积层和2个池化层交替的普通结构
    2. 3个Inception模块组串行
    3. 每个Inception模块组中又分不同的Inception Module串行
    4. 每个Inception Module的不同分支之间是并行，最后通过tf.concat在输出通道维度上合并
  '''

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    # 对[]中的三个函数设置默认值,在设置默认值的同时进行操作
    with slim.arg_scope(
      [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
      stride = 1,
      padding = 'VALID'):
      # 非Inception模块组的卷积层
      # slim.conv2d(input_tensor, output_channels, kernel_size, strides, padding, scope)
      # 假设输入图片尺寸为299 * 299，则经过三次stride=2的压缩后，图片大小为35*35
      net = slim.conv2d(inputs, 32, [3,3], stride=2, scope='Conv2d_1a_3*3')
      net = slim.conv2d(net, 32, [3,3], scope='Conv2d_2a_3*3')
      net = slim.conv2d(net, 64, [3,3], padding='SAME', scope='Conv2d_2b_3*3')
      net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_3a_3*3')
      net = slim.conv2d(net, 80, [1,1], scope='Conv2d_3b_1*1')
      net = slim.conv2d(net, 192, [3,3], scope='Conv2d_4a_3*3')
      net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_5a_3*3')

    # 模块组部分。Inception V3有三个连续的模块组,每个模块组中有多个Inception Module

    with slim.arg_scope(
      [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
      stride = 1,
      padding = 'SAME'):
      # ----------------------------第一个模块组----------------------------
      # 第一个模块组的第一个Inception Module - Mixed_5b
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5*5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3*3')
          branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3*3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 32, [1,1], scope='Conv2d_0b_1*1')
        # 合并4个分支的输出，3标识在第3个维度上合并，即输出通道数，4个分支的输出通道数之和为64+64+96+32=256
        # output - 35 * 35 * 256
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 第一个模块组的第二个Inception Module - Mixed_5c
      with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5*5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3*3')
          branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3*3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_1*1')   
        # 输出通道数为64+64+96+96=288
        # output - 35 * 35 * 288
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)       

      # 第一个模块组的第三个Inception Module - Mixed_5d
      with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5*5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3*3')
          branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3*3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_1*1')
        # 输出通道数为64+64+96+64=288
        # output - 35 * 35 * 288
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3) 

      # ----------------------------第二个模块组----------------------------
      # 第二个模块组的第一个Inception Module - Mixed_6a
      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 384, [3,3], stride=2, padding='VALID', scope='Conv2d_1a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 96, [3,3], scope='Conv2d_0b_3*3')
          branch_1 = slim.conv2d(branch_1, 96, [3,3], stride=2, padding='VALID', scope='Conv2d_1a_1*1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3,3], stride=2, padding='VALID', scope='MaxPool_1a_3*3')
        # output = 17 * 17 * (384+96+288) = 17*17*768
        net = tf.concat([branch_0, branch_1, branch_2], 3)

      # 第二个模块组的第二个Inception Module - Mixed_6b
      with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 128, [1,7], scope='Conv2d_0b_1*7')
          branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0b_7*1')
          branch_2 = slim.conv2d(branch_2, 128, [1,7], scope='Conv2d_0c_1*7')
          branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0d_7*1')
          branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
        # output - 17 * 17 * (192 + 192 + 192 + 192) = 17 * 17 * 768
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 第二个模块组的第三个Inception Module - Mixed_6c
      with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0b_1*7')
          branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0b_7*1')
          branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0c_1*7')
          branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0d_7*1')
          branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
        # output - 17 * 17 * (192 + 192 + 192 + 192) = 17 * 17 * 768
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 第二个模块组的第四个Inception Module - Mixed_6d
      with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0b_1*7')
          branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0b_7*1')
          branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0c_1*7')
          branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0d_7*1')
          branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
        # output - 17 * 17 * (192 + 192 + 192 + 192) = 17 * 17 * 768
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 第二个模块组的第五个Inception Module - Mixed_6e
      with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0b_1*7')
          branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 192, [7,1], scope='Conv2d_0b_7*1')
          branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0c_1*7')
          branch_2 = slim.conv2d(branch_2, 192, [7,1], scope='Conv2d_0d_7*1')
          branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1*7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
        # output - 17 * 17 * (192 + 192 + 192 + 192) = 17 * 17 * 768
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 将Mixed_6e存入end_points中
      end_points['Mixed_6e'] = net

      # ----------------------------第三个模块组----------------------------
      # 第三个模块组的第一个Inception Module - Mixed_7a
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
          branch_0 = slim.conv2d(branch_0, 320, [3,3], stride=2, padding='VALID' scope='Conv2d_0b_3*3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0b_1*7')
          branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7*1')
          branch_1 = slim.conv2d(branch_1, 192, [3,3], stride=2, padding='VALID' scope='Conv2d_0c_7*1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3,3], stride=2, padding='VALID', scope='MaxPool_1a_3*3')
        # output = 8 * 8 * (320 + 192 + 768) = 8 * 8 * 1280
        net = tf.concat([branch_0, branch_1, branch_2], 3)

      # 第三个模块组的第二个Inception Module - Mixed_7b
      with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = tf.concat([
            slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1*3'),
            slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_0c_3*1')
          ], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 384, [3,3], scope='Conv2d_0b_3*3')
          branch_2 = tf.concat([
            slim.conv2d(branch_2, 384, [1,3], scope='Conv2d_0c_1*3'),
            slim.conv2d(branch_2, 384, [3,1], scope='Conv2d_0d_3*1')
          ], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
        # output = 8 * 8 * (320 + 384*2 + 384*2 + 192) = 8 * 8 * 2048
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

      # 第三个模块组的第三个Inception Module - Mixed_7c
      with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1*1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1*1')
          branch_1 = tf.concat([
            slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1*3'),
            slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_0c_3*1')
          ], 3)
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, 448, [1,1], scope='Conv2d_0a_1*1')
          branch_2 = slim.conv2d(branch_2, 384, [3,3], scope='Conv2d_0b_3*3')
          branch_2 = tf.concat([
            slim.conv2d(branch_2, 384, [1,3], scope='Conv2d_0c_1*3'),
            slim.conv2d(branch_2, 384, [3,1], scope='Conv2d_0d_3*1')
          ], 3)
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3*3')
          branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1*1')
        # output = 8 * 8 * (320 + 384*2 + 384*2 + 192) = 8 * 8 * 2048
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


      
      return net, end_points

# 全局平均池化
'''
  num_classes -> 最终需要分类的数量
  is_training -> 是否是训练过程，Batch Normalization和Dropout只在is_training=true时启用
  dropout_keep_prob -> dropout时所需保留的节点比例
  prediction_fn -> 分类函数
  spatial_squeeze -> 是否对输出进行squeeze操作，去除维数为1的维度
  reuse -> 是否对网络和variable进行重复使用
  scope -> 包含了函数默认参数的环境

'''
def inception_V3(inputs,
  num_classes=1000,
  is_training=True,
  dropout_keep_prob=0.8,
  prediction_fn=slim.softmax,
  spatial_squeeze=True,
  reuse=None,
  scope='InceptionV3'):
  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
      net, end_points = inception_v3_base(inputs, scope)

  # 辅助分类节点 - Auxiliary Logits
  with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
    # 17*17*768
    aux_logits = end_points['Mixed_6e']
    with tf.variable_scope('AuxLogits'):
      aux_logits = slim.avg_pool2d(aux_logits, [5,5], stride=3, padding='VALID', scope='AvgPool_1a_5*5')  # 5*5*768
      aux_logits = slim.conv2d(aux_logits, 128, [1,1], scope='Conv2d_1b_1*1')   # 5*5*128
      aux_logits = slim.conv2d(aux_logits, 768, [5,5], weights_initializer=trunc_normal(0.01), padding='VALID', scope='Conv2d_2a_5*5')    # 1*1*768
      aux_logits = slim.conv2d(aux_logits, num_classes, [1,1], activation_fn=None, normalizer_fn=None, weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1*1')

      if spatial_squeeze:
        aux_logits = tf.squeeze(aux_logits, [1,2], name='SpatialSqueeze')

      end_points['AuxLogits'] = aux_logits

  # 正常结果预测
  with tf.variable_scope('Logits'):
    net = slim.avg_pool2d(net, [8,8], padding='VALID', scope='AvgPool_1a_8*8')
    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
    end_points['PreLogits'] = net
    logits = slim.conv2d(net, num_classes, [1,1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1*1')

    if spatial_squeeze:
      logits = tf.squeeze(logits, [1,2], name='SpatialSqueeze')
  
  end_points['Logits'] = logits
  end_points['Predictions'] = prediction_fn(logits, scope='Prediction')

  return logits, end_points




