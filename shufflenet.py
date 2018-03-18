# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
import config
from go import MAX_BOARD, N, BORDER

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

def max_pool(x):
    return tf.nn.max_pool(x, [1,1,2,2], strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')

def batchnorm(x, training):
    return tf.layers.batch_normalization(x, epsilon=1e-5, axis=1, fused=True,
                    center=False, scale=False, training=training)

class ShuffleNet():
    def __init__(self, filters=64, net_type="resnet", training=True, filter_size=3, input_size=19, input_planes=2):
        self.filters=filters
        self.net_type=net_type
        self.input_size=input_size
        self.input_planes = input_planes
        self.batch_norm_count = 0
        self.training = training
        self.filter_size=filter_size
        self.weights = []
        self.groups=4

    def channel_shuffle(self, net):
        net = tf.split(net, self.filters, axis=1, name="split")
        chs = []
        group_channels = self.filters // self.groups
        for i in range(self.groups):
            for j in range(group_channels):
                chs.append(net[i + j * self.groups])
        net = tf.concat(chs, axis=1, name="concat")
        return net

    def get_batchnorm_key(self):
        result = "bn" + str(self.batch_norm_count)
        self.batch_norm_count += 1
        return result

    def out_block(self, inputs, filter_size, input_channels, output_channels):
        W_conv = l2_weight_variable([filter_size, filter_size,
                                  input_channels, output_channels])
        b_conv = bias_variable([output_channels])
        self.weights.append(W_conv)
        self.weights.append(b_conv)
        weight_key = self.get_batchnorm_key()
        #self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        #self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            net = tf.nn.conv2d(inputs, W_conv, [1,1,1,1], "VALID", data_format="NCHW")
            net = tf.nn.bias_add(net, b_conv, data_format="NCHW")
            #net = batchnorm(net, training=self.training)
        #net = tf.nn.relu(net)
        return net

    def pool_block(self, inputs, filter_size, input_channels, output_channels, padding="SAME", stride=1):
        W_conv = l2_weight_variable([filter_size, filter_size,
                                  input_channels, output_channels])
        self.weights.append(W_conv)
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            net = tf.nn.conv2d(inputs, W_conv, [1,1,stride,stride], padding=padding, data_format="NCHW")
            net = batchnorm(net, training=self.training)
        net = tf.nn.relu(net)
        return net

    def conv_block(self, inputs, filter_size, input_channels, output_channels, relu=True):
        W_conv = l2_weight_variable([filter_size, filter_size,
                                  input_channels, output_channels])
        self.weights.append(W_conv)
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")

        with tf.variable_scope(weight_key):
            net = batchnorm(conv2d(inputs, W_conv), training=self.training)
        if relu:
            net = tf.nn.relu(net)
        return net
    
    def group_conv(self, inputs, filter_size, input_channels, output_channels, relu=True):
        groups = 4
        group_channels = input_channels // groups
        out_groups = output_channels // groups
        net = tf.split(inputs, groups, axis=1, name="split")
        for i in range(groups):
            net[i] = self.conv_block(net[i], filter_size, group_channels, out_groups, relu=False)
        net = tf.concat(net, axis=1, name="concat")
        if relu:
            net = tf.nn.relu(net)
        return net

    def shuffle_bottleneck(self, net, input_channels, output_channels):
        if input_channels!=output_channels:
            net_skip = self.conv_block(net, 1, input_channels, output_channels)
        else:
            net_skip = net
        pj = int(output_channels)
        net = self.group_conv(net, 1, input_channels, pj)
        net = self.channel_shuffle(net)
        depthwise_filter = l2_weight_variable([self.filter_size, self.filter_size, pj, 1])
        #b_conv = bn_bias_variable([self.filters])
        self.weights.append(depthwise_filter)
        #self.weights.append(b_conv)
        weight_key = self.get_batchnorm_key()
        self.weights.append(weight_key + "/batch_normalization/moving_mean:0")
        self.weights.append(weight_key + "/batch_normalization/moving_variance:0")
        #depthwise_filter = tf.get_variable("depth_conv_w", [3, 3, filters, 1],
        #                                   initializer=tf.truncated_normal_initializer(stddev=0.01))
        with tf.variable_scope(weight_key):
            net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, 1, 1, 1], 'SAME', data_format='NCHW', name="DWConv")
            net = batchnorm(net, training=self.training)
        net = self.group_conv(net,1, pj, output_channels, relu=False)

        #if 1 != stride:
        #    net = tf.concat([net, net_skip], axis=1)
        #else:
        net = tf.nn.relu(tf.add(net, net_skip))
        return net

    def bottleneck(self, net, input_channels, output_channels):
        if input_channels!=output_channels:
            net_skip = self.conv_block(net, 1, input_channels, output_channels)
        else:
            net_skip = tf.identity(net)
        net = self.conv_block(net, 1, input_channels, int(output_channels/4))
        net = self.conv_block(net, 3, int(output_channels/4), int(output_channels/4))
        net = self.conv_block(net, 1, int(output_channels/4), output_channels, relu=False)
        net = tf.nn.relu(tf.add(net, net_skip))
        return net

    def fuse_bottleneck(self, net1, net2, input_channels, output_channels):
        if input_channels!=output_channels:
            net_skip = self.conv_block(net1+net2, 1, input_channels, output_channels)
        else:
            net_skip = net1+net2
        net1 = self.conv_block(net1, 1, input_channels, int(output_channels/4))
        net1 = self.conv_block(net1, 3, int(output_channels/4), int(output_channels/4))
        net1 = self.conv_block(net1, 1, int(output_channels/4), output_channels, relu=False)
        net1 = tf.nn.relu(tf.add(net1, net_skip))
        net2 = self.conv_block(net2, 1, input_channels, int(output_channels/4))
        net2 = self.conv_block(net2, 3, int(output_channels/4), int(output_channels/4))
        net2 = self.conv_block(net2, 1, int(output_channels/4), output_channels, relu=False)
        net2 = tf.nn.relu(tf.add(net2, net_skip))
        return net1, net2

    def construct_net(self, planes, blocks):
        pol_channel=8
        val_channel=1
        pad = int((MAX_BOARD-self.input_size)/2+1)
        inputs = tf.reshape(planes, [-1, self.input_planes, self.input_size, self.input_size])
        print("old input: ", inputs.get_shape().as_list(), file=sys.stderr)
        inputs = tf.pad(inputs, [[0,0],[0,0],[pad,pad],[pad,pad]], constant_values = BORDER)
        print("input shape: ", inputs.get_shape().as_list(), file=sys.stderr)
        # Input convolution
        flow = inputs
        if self.net_type != "resnet":
            flow = self.conv_block(inputs, 3, self.input_planes, self.filters)
        print("after input: ", flow.get_shape().as_list(), file=sys.stderr)
        #flow = tf.pad(flow, [[0,0],[0,0],[0,1],[0,1]])
        #print("after pad: ", flow.get_shape().as_list(), file=sys.stderr)
        # Residual tower 11*11 16
        b1 = int(blocks*0.3)
        b3 = max(blocks-b1-1, min(1, int(blocks*0.3)))
        b2 = blocks-b1-b3
        tower_out = self.filters*4
        if self.net_type=="fractal":
            pol_channel=16
            val_channel=1
            tower_out = self.filters*3
            for _ in range(b1):
                flow = self.bottleneck(flow, self.filters, self.filters)
            flow = self.pool_block(flow, 2, self.filters, self.filters*2, stride=2)
            for _ in range(b2):
                flow = self.bottleneck(flow, self.filters*2, self.filters*2)
            flow = self.pool_block(flow, 2, self.filters*2, tower_out, stride=2)
            for _ in range(b3):
                flow = self.bottleneck(flow, tower_out, tower_out)
        elif self.net_type=="shuffle":
            for _ in range(b1):
                flow = self.shuffle_bottleneck(flow, self.filters, self.filters)
            flow = self.pool_block(flow, 3, self.filters, self.filters*2)
            for _ in range(b2):
                flow = self.shuffle_bottleneck(flow, self.filters*2, self.filters*2)
            flow = self.pool_block(flow, 3, self.filters*2, tower_out, stride=1)
            for _ in range(b3):
                flow = self.shuffle_bottleneck(flow, tower_out, tower_out)
        elif self.net_type == "fuse":
            flows = tf.split(flow, 2, axis=1, name="split")
            flow1=flows[0]
            flow2=flows[1]
            print("split: ", flow1.get_shape().as_list())
            filters= int(self.filters/2)
            for _ in range(b1):
                flow1, flow2 = self.fuse_bottleneck(flow1, flow2, filters, filters)
            flow1 = self.pool_block(flow1, 3, filters, filters*2)
            flow2 = self.pool_block(flow2, 3, filters, filters*2)
            for _ in range(b2):
                flow1, flow2 = self.fuse_bottleneck(flow1, flow2, filters*2, filters*2)
            flow1 = self.pool_block(flow1, 3, filters*2, tower_out, stride=1)
            flow2 = self.pool_block(flow2, 3, filters*2, tower_out, stride=1)
            for _ in range(b3):
                flow1, flow2 = self.fuse_bottleneck(flow1, flow2, tower_out, filters*4)
            flow = tf.concat([flow1,flow2], axis=1)
        else:
            pol_channel=16
            val_channel=1
            #b4 = blocks-7
            flow = self.conv_block(flow, 3, self.input_planes, self.filters)
            for _ in range(b1):
                flow = self.residual_block(flow, self.filters, self.filters)
            flow = self.pool_block(flow, 2, self.filters, self.filters*2, stride=2)
            for _ in range(b2):
                flow = self.residual_block(flow, self.filters*2, self.filters*2)
            flow = self.pool_block(flow, 2, self.filters*2, tower_out, stride=2)
            for _ in range(b3):
                flow = self.residual_block(flow, tower_out, tower_out)
            #flow = self.pool_block(flow, 3, self.filters*3, self.filters*4, padding="VALID", stride=1)
            #for _ in range(b4):
            #    flow = self.residual_block(flow, self.filters*4, self.filters*4)
        print("after tower: ", flow.get_shape().as_list(), file=sys.stderr)

        # Policy head
        shape =flow.get_shape().as_list()
        size = shape[2]*shape[3]
        pol_net = self.conv_block(flow, 1, tower_out, pol_channel)
        pol_net = tf.reshape(pol_net, [-1, pol_channel*size])
        #print(pol_net.get_shape().as_list())
        W_fc1 = l2_weight_variable([pol_channel*size, config.policy_size])
        b_fc1 = bias_variable([config.policy_size])
        self.weights.append(W_fc1)
        self.weights.append(b_fc1)
        with tf.variable_scope("pol_fc"):
            pol_net = tf.add(tf.matmul(pol_net, W_fc1), b_fc1)
        print("policy head: ", pol_net.get_shape().as_list(), file=sys.stderr)
        #pol_net = tf.reshape(pol_net, [-1, pis])

        # Value head
        val_net = self.conv_block(flow, 1,tower_out, val_channel)
        val_net = tf.reshape(val_net, [-1,val_channel*size])
        W_fc2 = l2_weight_variable([val_channel*size, 1])
        b_fc2 = bias_variable([1])
        self.weights.append(W_fc2)
        self.weights.append(b_fc2)
        with tf.variable_scope("val_fc"):
            val_net = tf.add(tf.matmul(val_net, W_fc2), b_fc2)
        val_net = tf.nn.tanh(val_net)   #*self.input_size*self.input_size
        val_net = tf.reshape(val_net, [-1,])
        print("value head: ", val_net.get_shape().as_list(), file=sys.stderr)

        return pol_net, val_net

    def residual_block(self, net, input_channels, output_channels):
        if input_channels!=output_channels:
            net_skip = self.pool_block(net, 3, input_channels, output_channels)
        else:
            net_skip = tf.identity(net)
        net = self.conv_block(net_skip, 3, output_channels, output_channels)
        # Second convnet
        net = self.conv_block(net, 3, output_channels, output_channels, relu=False)
        net = tf.nn.relu(tf.add(net, net_skip))
        return net

    
'''
        net = shuffle_stage(net, base_ch, 3, groups, 'Stage2')
        net = shuffle_stage(net, base_ch * 2, 7, groups, 'Stage3')
        net = shuffle_stage(net, base_ch * 4, 3, groups, 'Stage4')
    def shuffle_stage(net, output, repeat, group, scope="Stage"):
        with tf.variable_scope(scope):
            net = shuffle_bottleneck(net, output, 2, group, scope='Unit{}'.format(0))
            for i in range(repeat):
                net = shuffle_bottleneck(net, output, 1, group, scope='Unit{}'.format(i + 1))
        return net
'''
