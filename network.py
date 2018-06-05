#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#

# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

import numpy as np
import time
import tensorflow as tf
import config
import go
import dataset
import struct
import sys
import net

class Network:
    def __init__(self, height, is_train=False):
        self.time_start = time.time()
        self.save_file = '%s/%s' % (config.save_dir, config.save_name)
        self.height = height
        self.is_train = is_train
        self.gragh=tf.Graph()
        with self.gragh.as_default():
            self.set_network()

    def set_network(self):
        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.x = tf.placeholder(tf.float32, [None, 3, dataset.IMAGE_SIZE, dataset.IMAGE_SIZE])
        nt = net.ResNet(config.RESIDUAL_FILTERS, self.training, 3, dataset.IMAGE_SIZE, 3)
        self.y, self.weights = nt.construct_net(self.x, self.height)

        if self.is_train:
            self.y_ = tf.placeholder(tf.float32, [None, 4])
            # Loss on value head
            self.mse_loss = tf.reduce_sum(tf.reduce_mean(tf.squared_difference(self.y_, self.y), axis=0))

            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            loss = self.mse_loss + self.l2_loss

            boundaries = [200000, 400000, 600000, 1000000]
            values = [config.learning_rate, config.learning_rate*0.1, config.learning_rate*0.01,
                      config.learning_rate*0.001, config.learning_rate*0.0005]
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            opt_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                momentum=config.momentum, use_nesterov=True)

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_op = opt_op.minimize(loss, global_step=self.global_step)

            self.avg_mse_loss = []
            self.avg_l2_loss = []

            # Summary part
            self.test_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), config.log_dir+"/test"), self.gragh)
            self.train_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), config.log_dir+"/train"), self.gragh)

        self.saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        cfg = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(graph=self.gragh, config=cfg)
        self.restore()
        ts = time.time()-self.time_start
        config.logger.info('%d分%.2f秒：网络构造及参数初始化完成，开始博弈……' % (ts//60, ts % 60))

    def restore(self, file=None):
        with self.gragh.as_default():
            if not file:
                file = self.save_file
            if os.path.isfile(file+'.meta'):
                self.saver.restore(self.session, file)
                #path = os.path.dirname(file)
                #self.restore_elo(file)
                config.logger.info("参数已从{}载入，轮数：{} ".format(file, self.get_step()))
            else:
                self.session.run(tf.global_variables_initializer())

    def get_step(self):
        with self.gragh.as_default():
            return self.session.run(self.global_step)

    def train(self, train_data, batch_size=0):
        if batch_size==0:
            batch_size = config.batch_size
        if not self.time_start:
            self.time_start = time.time()
        while config.running and (train_data.isloading or train_data.data_size>batch_size):
            if train_data.data_size<batch_size:
                time.sleep(5)
                continue
            img, label = train_data.get_batch(batch_size)
            steps, mse_loss, l2_loss, _ = self.session.run([self.global_step, 
                self.mse_loss, self.l2_loss, self.train_op],
                feed_dict={self.training: True, self.x:img, self.y_:label})
            mse_loss = mse_loss / 4.0
            self.avg_mse_loss.append(mse_loss)
            self.avg_l2_loss.append(l2_loss)
            if steps % 100 == 0:
                time_end = time.time()
                speed = 0
                if self.time_start:
                    elapsed = time_end - self.time_start
                    speed = batch_size * (100.0 / elapsed)
                avg_mse_loss = np.mean(self.avg_mse_loss or [0])
                avg_l2_loss = np.mean(self.avg_l2_loss or [0])
                config.logger.info("step {}, 总差={:.2f}，价值={:.2f}，l2={:.2f}，({:.1f} pos/s)".format(
                    steps, 4.0 * avg_mse_loss+l2_loss,avg_mse_loss,l2_loss,speed))
                lr = self.session.run(self.learning_rate)
                train_summaries = tf.Summary(value=[
                    tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss),
                    tf.Summary.Value(tag="L2 Loss", simple_value=avg_l2_loss),
                    tf.Summary.Value(tag="Learning Rate", simple_value=lr)])
                self.train_writer.add_summary(train_summaries, steps)
                self.time_start = time_end
                self.avg_mse_loss = []
                self.avg_l2_loss = []
                self.save(self.save_file)
            # Ideally this would use a seperate dataset and so on...
            if steps % 1000 == 0:
                sum_mse = 0
                test_batches = 100
                acc = 0.0
                for _ in range(test_batches):
                    while train_data.data_size<batch_size:
                        time.sleep(5)
                    img, label = train_data.get_batch(batch_size)
                    test_mse,vs = self.session.run([self.mse_loss, self.y], feed_dict={self.training: False, 
                           self.x:img, self.y_:label})
                    for j in range(batch_size):
                        for i in range(4):
                            acc += 1 if abs(vs[j][i]-label[j][i])/label[j][i]<0.01 else 0
                    sum_mse += test_mse
                acc /= test_batches*4
                sum_mse /= (4.0 * test_batches)
                test_summaries = tf.Summary(value=[
                    tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse),
                    tf.Summary.Value(tag="Acc ", simple_value=acc)])
                self.test_writer.add_summary(test_summaries, steps)
                config.logger.info("step:{}, mse={:.2f}, acc={:.2f}".format(steps, sum_mse, acc))
                self.save(self.save_file)
        config.logger.info("isloading: {}, config.running: {}".format(train_data.isloading, config.running))
        self.save(self.save_file)
        config.logger.info("权重保存完毕: ", steps)
        #self.save_variables()

    def run(self, img):
        datas = self.run_many([img])
        return datas[0]

    def run_many(self, imgs):
        with self.gragh.as_default():
            datas, = self.session.run([self.y],
                feed_dict={self.training: False, self.x: imgs})
            config.logger.info("many datas:",datas)
            datas[:][0] *=go.MAX_BOARD
            datas[:][1:] *=self.height
        return datas
       
    def save(self, fn):
        if fn is None or not self.is_train:
            return
        path = os.path.dirname(fn)
        if not os.path.exists(path):
            os.makedirs(path)
        with self.gragh.as_default():
            self.saver.save(self.session, fn)
            #self.save_leelaz_weights(path+"zigo.txt")
            #self.save_elo(fn)

    def save_variables(self, fn=None):
        if not fn:
            fn = self.save_file
        if fn is not None:
            path = os.path.dirname(fn)
            pathold = path+'/back/'
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(pathold):
                os.makedirs(pathold)
            pathold += config.save_name
            with self.gragh.as_default():
                steps = self.session.run(self.global_step)
                save_path = self.saver.save(self.session, pathold, global_step=steps)
                self.save_elo(save_path)
                config.logger.info("Model saved in file: {}".format(save_path))

    def save_leelaz_weights(self, filename):
        bf = open(filename.replace("txt", "bid"), "wb")
        if config.policy_size==go.MAX_BOARD*go.MAX_BOARD+2:
            bf.write(struct.pack('i',4))
        else:
             bf.write(struct.pack('i',3))           
        bf.write(struct.pack('i',config.BLOCKS_NUM))
        bf.write(struct.pack('i',config.FEATURE_NUM))
        bf.write(struct.pack('i',config.RESIDUAL_FILTERS))
        bf.write(struct.pack('i',go.MAX_BOARD))
        with open(filename, "w") as file:
            # Version tag
            file.write("1")
            for weights in self.net.weights:
                # Newline unless last line (single bias)
                file.write("\n")
                work_weights = None
                # Keyed batchnorm weights
                if isinstance(weights, str):
                    work_weights = tf.get_default_graph().get_tensor_by_name(weights)
                elif weights.shape.ndims == 4:
                    # Convolution weights need a transpose
                    #
                    # TF (kYXInputOutput)
                    # [filter_height, filter_width, in_channels, out_channels]
                    #
                    # Leela/cuDNN/Caffe (kOutputInputYX)
                    # [output, input, filter_size, filter_size]
                    work_weights = tf.transpose(weights, [3, 2, 0, 1])
                elif weights.shape.ndims == 2:
                    # Fully connected layers are [in, out] in TF
                    #
                    # [out, in] in Leela
                    #
                    work_weights = tf.transpose(weights, [1, 0])
                else:
                    # Biases, batchnorm etc
                    work_weights = weights
                nparray = work_weights.eval(session=self.session)
                for w in nparray.flatten():
                    bf.write(struct.pack('f',w))
                wt_str = [str(wt) for wt in np.ravel(nparray)]
                file.write(" ".join(wt_str))
        bf.close()
        config.logger.info("保存完毕。")
                
    def save_weights_bin(self, fn, txtfn):
        txt = open(txtfn, "r")
        bf = open(fn, "wb")
        if config.groups>0:
            bf.write(struct.pack('i',4))
        else:
             bf.write(struct.pack('i',3))           
        bf.write(struct.pack('i',config.BLOCKS_NUM))
        bf.write(struct.pack('i',config.FEATURE_NUM))
        bf.write(struct.pack('i',config.RESIDUAL_FILTERS))
        if config.groups>0:
            bf.write(struct.pack('i',config.input_pad))
        bf.write(struct.pack('i',go.N))
        line = txt.readline()
        i = 0
        while line:
            line = txt.readline()
            i+=1
            ms = line.split(" ") #re.findall(r"([-]?[0-9\.]+?)[ |\n]", line)
            if ms:
                for s in ms:
                    if s and s!="\n" and s!="\r\n":
                        w=float(s)
                        bf.write(struct.pack('f',w))

        txt.close()
        bf.close()
        config.logger.info("保存完毕。")

