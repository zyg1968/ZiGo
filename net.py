#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
import config
import dataset
from PIL import Image

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def l2_weight_variable(shape):
    stddev = np.sqrt(2.0 / (sum(shape)))
    initial = tf.truncated_normal(shape, stddev=stddev)
    weights = tf.Variable(initial)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
    return weights

# Bias weights for layers not followed by BatchNorm
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

# No point in learning bias weights as they are cancelled
# out by the BatchNorm layers's mean adjustment.
def bn_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, trainable=False)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

def conv2d_s2(x, W):
    return tf.nn.conv2d(x, W, [1,1,2,2], padding='SAME', data_format="NCHW")

def batchnorm(x, training):
    return tf.layers.batch_normalization(x, epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False, training=training)

def max_pool(x):
    return tf.nn.max_pool(x, [1,1,2,2], strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')

class ResNet():
    def __init__(self, filters=32, training=True, filter_size=3, input_size=224, input_planes=3):
        self.filters=filters
        self.input_size=input_size
        self.input_planes = input_planes
        self.batch_norm_count = 0
        self.training = training
        self.filter_size=filter_size
        self.weights = []

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def pool_block(self, flow, filter_size, input_channels, output_channels, padding="SAME", stride=1):
        W_conv = l2_weight_variable([filter_size, filter_size,
                                    input_channels, output_channels])
        self.weights.append(W_conv)
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            flow = tf.nn.conv2d(flow, W_conv, [1,1,stride,stride], padding=padding, data_format="NCHW")
            flow = batchnorm(flow, training=self.training)
        flow = tf.nn.relu(flow)
        return flow

    def conv_block(self, flow, filter_size, input_channels, output_channels, relu=True):
        W_conv = l2_weight_variable([filter_size, filter_size,
                                    input_channels, output_channels])
        self.weights.append(W_conv)
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            flow = batchnorm(conv2d(flow, W_conv), training=self.training)
        if relu:
            flow = tf.nn.relu(flow)
        return flow

    def bottleneck(self, flow, input_channels, output_channels):
        if input_channels!=output_channels:
            net_skip = self.conv_block(flow, 1, input_channels, output_channels)
        else:
            net_skip = tf.identity(flow)
        flow = self.conv_block(flow, 1, input_channels, int(output_channels/4))
        flow = self.conv_block(flow, 3, int(output_channels/4), int(output_channels/4))
        flow = self.conv_block(flow, 1, int(output_channels/4), output_channels, relu=False)
        flow = tf.nn.relu(tf.add(flow, net_skip))
        return flow

    def pool_bottleneck(self, flow, input_channels, output_channels):
        flow = self.conv_block(flow, 1, input_channels, output_channels)
        flow = self.pool_block(flow, 3, output_channels, output_channels, stride=2)
        #flow = self.conv_block(flow, 1, int(output_channels), output_channels)
        return flow

    def construct_net(self, img, height):
        out_size = 4
        #统一输入
        #padx = int((max_width-width)/2)
        #pady = int((max_height-height)/2)
        flow = tf.reshape(img, [-1, self.input_planes, self.input_size, self.input_size])
        config.logger.info("old input: {}".format(flow.get_shape().as_list()))
        flow = self.conv_block(flow, self.filter_size, self.input_planes, self.filters)
        flow = max_pool(flow)
        shape = flow.get_shape().as_list()
        config.logger.info("after conv1: {}".format(shape))  #150*150*64
        for i in range(2):
            flow = self.bottleneck(flow, shape[1], shape[1]*2) #64/128
            flow = max_pool(flow)
            shape = flow.get_shape().as_list()
            config.logger.info("after conv{}: {}".format(i+1, shape)) #75*75*64/38*38*128
        #第一个输出 
        #print("resnet out: ", shape, file=sys.stderr)
        classes=[]
        bboxs=[]
        cls = self.conv_block(flow, shape[1], 1)    #是否棋子
        bbox = self.conv_block(flow, shape[1], 4*1)  #棋子位置和大小
        classes.append(cls)
        bboxs.append(bbox)
        for i in range(4):
            flow = self.pool_bottleneck(flow, shape[1], shape[1]/2) #256/128/64/32
            #flow = max_pool(flow)
            shape = flow.get_shape().as_list()
            config.logger.info("after conv{}: {}".format(i+1, shape)) #19/9/5/3
        #3*3*512
        cls = self.conv_block(flow, shape[1], 1)    #是否棋盘
        bbox = self.conv_block(flow, shape[1], 4*1)  #棋盘位置和大小
        classes.append(cls)
        bboxs.append(bbox)

        '''        flow = self.conv_block(flow, 1, shape[1], 1)
        print("after out conv: ", flow.get_shape().as_list(), file=sys.stderr) #7*7*1
        shape = flow.get_shape().as_list()
        data_size = shape[1]*shape[2]*shape[3]
        flow = tf.reshape(flow, [-1, data_size])
        W_fc1 = l2_weight_variable([data_size, out_size])
        b_fc1 = bias_variable([out_size])
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        with tf.variable_scope("fc1"):
            flow = tf.add(tf.matmul(flow, W_fc1), b_fc1)
        W_fc2 = l2_weight_variable([midlle_size, out_size])
        b_fc2 = bias_variable([out_size])
        weights.append(W_fc2)
        weights.append(b_fc2)
        with tf.variable_scope("fc2"):
            flow = tf.add(tf.matmul(flow, W_fc2), b_fc2)'''
        return classes, bboxs, self.weights
    