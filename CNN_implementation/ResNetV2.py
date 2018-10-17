#!/usr/bin/python
# -*- coding:utf8 -*-

'''
  ResNet V2 结构

'''

import collections
import tensorflow as tf
slim = tf.contrib.slim

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  'A named tuple describing a ResNet block.'

'''
  Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)])
          |           |                             |
          |_ scope    |                             |
                      |_ResNet V2中残差学习单元       |
                        每个残差学习单元包含3个卷积层    |
                                                    |_block的args
  block的args解析，比如[(256, 64, 3)]
    第三层输出通道数256
    前两层输出通道数64
    中间层步长为3 
  这个残差学习单元结构即为[(1*1/s1, 64), (3*3/s2, 64), (1*1/s1, 256)] 

这TMD什么意思啊                                                 
'''

# 降采样，factor为1则返回输入，否则执行1*1的最大池化，步长为传入的factor
def subsample(inputs, factor, scope=None):
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1,1], stride=factor, scope=scope)

# 创建卷积层, stride为1则直接执行slim.conv2d并设置padding='SAME'
# 否则设置相关pad操作后再执行卷积
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
  else:
    # 这个pad又是什么鬼，貌似是一种padding的方式？
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pag_beg
    inputs = tf.pad(inputs, [[0,0], [pad_beg, pad_end], [pad_beg, pad_end], [0,0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)

# 定义堆叠blocks的函数
'''
  net -> 输入
  blocks -> Block的class列表
  outputs_collections -> 收集各个end_points的collections
'''
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        with tf.variable_scope('unit_%d', %(i+1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit
          # block.unit_fn是什么，后续使用时定义？
          net = block.unit_fn(net,
            depth = unit_depth
            depth_bottleneck = unit_depth_bottleneck
            stride = unit_stride
          )
          net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

# 定义函数的参数默认值
def resnet_arg_scope(is_Training=True,
  weight_decay=0.0001,
  batch_norm_decay=0.997,
  batch_norm_epsilon=1e-5,
  batch_norm_scale=True):

  batch_norm_params = {
    'is_Training': is_Training,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
    [slim.conv2d],
    weights_regularizer = slim.l2_regularizer(weight_decay),
    weights_initializer = slim.variance_scaling_initializer(),
    activation_fn = tf.nn.relu,
    normalizer_fn = slim.batch_norm,
    normalizer_params = batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc

# 定义bottleneck残差学习单元
'''
  在每一层前均使用了Batch Normalization
  对输入进行了preactivation
'''
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    # 获取输入的最后一个维度，即通道数
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)

    # 对输入进行Batch Normalization,并使用ReLU执行预激活preactivation
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

    # 定义shortcut，即直连的x
    if depth == depth_in:
      # 如果输入的通道数（depth_in)和输出通道数（depth)一致，使用降采样
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      # 否则，执行1*1且步长为stride的卷积，设定输出通道数为depth
      shortcut = slim.conv2d(preact, depth, [1,1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')

    # 定义残差，有3层
    # 1. 1*1,步长为1,输出通道数depth_bottleneck
    residual = slim.conv2d(preact, depth_bottleneck, [1,1], stride=1, scope='conv1')
    # 2. 3*3, 步长为3,输出通道数depth_bottleneck
    residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
    residual = slim.conv2d(residual, depth, [1,1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

# 定义生成ResNet V2的主函数
'''
  inputs -> 输入
  num_classes -> 最后输出的类数
  global_pool -> 是否加上最后一层全局平均池化
  include_root_block -> 是否加上ResNet网络最前面通常使用的7*7卷积和最大池化
  reuse -> 是否重用
  scope -> 整个网络的名称
'''
def resnet_v2(inputs,
  blocks,
  num_classes=None,
  global_pool=True,
  include_root_block=True,
  reuse=None,
  scope=None):

  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'

    # 设定默认参数
    with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
      net = inputs
      # 如果设置为true,手动加上前面的7*7卷积和3*3最大池化
      if include_root_block:
        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
          net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
        net = slim.max_pool2d(net, [3,3], stride=2, scope='pool1')

      # 生成残差学习模块组
      net = stack_blocks_dense(net, blocks)
      net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

      # 根据标记判断是否添加全局平均池化
      if global_pool:
        # reduce_mean实现全局平均池化效率高于avg_pool2d?
        net = tf.reduce_mean(net, [1,2], name='pool5', keep_dims=True)
      
      # 如果传入输出分类数，添加一个输出通道数为num_classes的1*1卷积，无激活函数和正则项
      if num_classes is not None:
        net = slim.conv2d(net, num_classes, [1,1], activation_fn=None, normalizer_fn=None, scope='logits')
      
      # 转化collection为python的dict
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      # 添加softmax层输出网络结果
      if num_classes is not None:
        end_points['predictions'] = slim.softmax(net, scope='predictions')

      return net, end_points

# 50层的ResNet V2
def resnet_v2_50(inputs,
  num_classes=None,
  global_pool=True,
  reuse=None,
  scope='resnet_v2_50'):
  blocks = [
    # 为什么结构是1*2 + 2*1的模式
    Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
    Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
    Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
    Block('block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

# 101层的ResNet V2
def resnet_v2_101(inputs,
  num_classes=None,
  global_pool=True,
  reuse=None,
  scope='resnet_v2_101'):
  blocks = [
    Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
    Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
    Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
    Block('block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

# 152层的ResNet V2
def resnet_v2_152(inputs,
  num_classes=None,
  global_pool=True,
  reuse=None,
  scope='resnet_v2_152'):
  blocks = [
    Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
    Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
    Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
    Block('block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

# 200层的ResNet V2
def resnet_v2_200(inputs,
  num_classes=None,
  global_pool=True,
  reuse=None,
  scope='resnet_v2_200'):
  blocks = [
    Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
    Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
    Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
    Block('block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)
