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
import shufflenet

class PolicyNetwork:
    def __init__(self, is_train=False, use_old=False, selftrain=False):
        self.time_start = time.time()
        if go.MAX_BOARD*go.MAX_BOARD>config.policy_size:
            config.policy_size = go.MAX_BOARD*go.MAX_BOARD+2       
        if use_old:
            self.save_file = '%s/b%df%d/old/%s' % (config.save_dir, config.BLOCKS_NUM, 
                                           config.FEATURE_NUM, config.save_name)
        else:
            self.save_file = '%s/b%df%d/%s' % (config.save_dir, config.BLOCKS_NUM, 
                                           config.FEATURE_NUM, config.save_name)
        
        self.is_train = is_train
        self.selftrain = selftrain
        self.game_num = 0
        self.elo = -5000.0
        self.loss_times=0
        self.train_n = 1
        self.gragh=tf.Graph()
        with self.gragh.as_default():
            self.set_network()

    def set_network(self):
        # TF variables
        #self.next_batch = next_batch
        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.x = tf.placeholder(tf.float32, [None, config.FEATURE_NUM, go.N, go.N])
        self.net= shufflenet.ShuffleNet(filters=config.RESIDUAL_FILTERS, net_type=config.net_type,
                             training=self.training, filter_size=3, input_size=go.N,
                             input_planes=config.FEATURE_NUM)
        self.y_conv, self.z_conv = self.net.construct_net(self.x, config.BLOCKS_NUM)

        if self.is_train:
            self.y_ = tf.placeholder(tf.float32, [None, config.policy_size])
            self.z_ = tf.placeholder(tf.float32, [None, 1])

            # Calculate loss on policy head
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
            self.policy_loss = tf.reduce_mean(cross_entropy)

            # Loss on value head
            self.mse_loss = tf.reduce_mean(tf.squared_difference(self.z_, self.z_conv))

            # Regularizer 1e-4 * tf.add_n([tf.nn.l2_loss(v)
            #                  for v in tf.trainable_variables() if not 'bias' in v.name and not '_fc' in v.name])
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            loss = self.policy_loss + self.mse_loss + self.l2_loss

            #self.learning_rate = tf.train.exponential_decay(1e-2, global_step, config.decay_steps, config.decay_rate)
            #self.learning_rate= tf.constant(config.learning_rate, dtype=tf.float32)
            boundaries = [2000000, 4000000, 6000000, 8000000]
            values = [config.learning_rate, config.learning_rate*0.1, config.learning_rate*0.01,
                      config.learning_rate*0.001, config.learning_rate*0.0005]
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
            opt_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                momentum=config.momentum, use_nesterov=True)

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_op = opt_op.minimize(loss, global_step=self.global_step)

            correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(correct_prediction)

            self.avg_policy_loss = []
            self.avg_mse_loss = []
            self.avg_l2_loss = []

            # Summary part
            logdir = "%s/b%df%d" % (config.log_dir, config.BLOCKS_NUM, config.FEATURE_NUM)
            self.test_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), logdir+"/test"), self.gragh)
            self.train_writer = tf.summary.FileWriter(
                os.path.join(os.getcwd(), logdir+"/train"), self.gragh)

        self.saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
        cfg = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(graph=self.gragh, config=cfg)
        self.restore()
        ts = time.time()-self.time_start
        print('%d分%.2f秒：网络构造及参数初始化完成，开始博弈……' % (ts//60, ts % 60), file=sys.stderr)

    def restore_elo(self, file=None):
        if file is None:
            file = self.save_file
            if file is None:
                return
        with open(file+'elo.txt', 'r') as f:
            lines = f.readlines()
            self.elo = float(lines[0])
            self.game_num = int(lines[1])
            self.train_n = int(lines[2])
            self.loss_times = int(lines[3])

    def restore(self, file=None):
        with self.gragh.as_default():
            if not file:
                file = self.save_file
            if os.path.isfile(file+'.meta'):
                self.saver.restore(self.session, file)
                path = os.path.dirname(file)
                self.restore_elo(file)
                print("参数已从{}载入，轮数：{} 等级分：{:.2f}，对局数：{}，文件号：{}，失败次数：{}。".format(file,
                    self.get_step(), self.elo, self.game_num, self.train_n, self.loss_times))
            else:
                self.session.run(tf.global_variables_initializer())


    def change_learning_rate(self, lr):
        self.learning_rate = tf.constant(lr, dtype=tf.float32)

    def get_step(self):
        with self.gragh.as_default():
            return self.session.run(self.global_step)

    def train(self, train_data, batch_size=0):
        if batch_size==0:
            batch_size = config.batch_size
        if not self.time_start:
            self.time_start = time.time()
        while config.running and (train_data.isloading or train_data.data_size>batch_size):
            states, move_vals, points = train_data.get_batch(batch_size)
            if states is None:
                time.sleep(0.2)
                continue
            steps, policy_loss, mse_loss, l2_loss, _ = self.session.run(
                [self.global_step, self.policy_loss, self.mse_loss, self.l2_loss, self.train_op],
                feed_dict={self.training: True, self.x:states, self.y_:move_vals, self.z_:points})
            # Keep running averages
            # Google's paper scales MSE by 1/4 to a [0, 1] range, so do the same to
            # get comparable values.
            mse_loss = mse_loss / 4.0
            self.avg_policy_loss.append(policy_loss)
            self.avg_mse_loss.append(mse_loss)
            self.avg_l2_loss.append(l2_loss)
            if steps % 100 == 0:
                time_end = time.time()
                speed = 0
                if self.time_start:
                    elapsed = time_end - self.time_start
                    speed = batch_size * (100.0 / elapsed)
                avg_policy_loss = np.mean(self.avg_policy_loss or [0])
                avg_mse_loss = np.mean(self.avg_mse_loss or [0])
                avg_l2_loss = np.mean(self.avg_l2_loss or [0])
                print("step {}, 总差={:.2f}，策略={:.2f}，价值={:.2f}, l2={:.2f} ({:.1f} pos/s)".format(
                    steps, avg_policy_loss + 4.0 * avg_mse_loss + avg_l2_loss,
                    avg_policy_loss, avg_mse_loss, avg_l2_loss, speed), file=sys.stderr)
                lr = self.session.run(self.learning_rate)
                train_summaries = tf.Summary(value=[
                    tf.Summary.Value(tag="Policy Loss", simple_value=avg_policy_loss),
                    tf.Summary.Value(tag="MSE Loss", simple_value=avg_mse_loss),
                    tf.Summary.Value(tag="L2 Loss", simple_value=avg_l2_loss),
                    tf.Summary.Value(tag="Learning Rate", simple_value=lr)])
                self.train_writer.add_summary(train_summaries, steps)
                self.time_start = time_end
                self.avg_policy_loss, self.avg_mse_loss, self.avg_reg_term = [], [], []
            # Ideally this would use a seperate dataset and so on...
            if steps % 5000 == 0:
                sum_accuracy = 0
                sum_mse = 0
                sum_policy = 0
                test_batches = 500
                for _ in range(test_batches):
                    states, move_vals, points = train_data.get_batch(batch_size)
                    if states is None:
                        time.sleep(0.2)
                        continue
                    test_policy, test_accuracy, test_mse = self.session.run(
                    [self.policy_loss, self.accuracy, self.mse_loss], feed_dict={self.training: False, 
                           self.x:states, self.y_:move_vals, self.z_:points})
                    sum_accuracy += test_accuracy
                    sum_mse += test_mse
                    sum_policy += test_policy
                sum_accuracy /= test_batches
                sum_policy /= test_batches
                # Additionally rescale to [0, 1] so divide by 4
                sum_mse /= (4.0 * test_batches)
                test_summaries = tf.Summary(value=[
                    tf.Summary.Value(tag="Accuracy", simple_value=sum_accuracy),
                    tf.Summary.Value(tag="Elo Score", simple_value=self.elo),
                    tf.Summary.Value(tag="Policy Loss", simple_value=sum_policy),
                    tf.Summary.Value(tag="MSE Loss", simple_value=sum_mse)])
                self.test_writer.add_summary(test_summaries, steps)
                print("step:{}, games:{}, 准确={:.1f}%, 策略={:.2f} mse={:.2f} Elo={:.2f}".format(
                    steps, self.game_num, sum_accuracy*100.0, sum_policy, sum_mse, self.elo))
                self.save(self.save_file)
            #if steps % config.test_steps == 0:
            #    self.save_variables()
            #    if self.selftrain:
            #        break
        print("isloading: ", train_data.isloading, "config.running: ", config.running)
        #if not config.running:
        self.save(self.save_file)
        self.save_variables()

    def run(self, board):
        vs, ps = self.run_many([board])
        return vs[0], ps[0]

    def run_many(self, boards):
        features = [dataset.get_feature(board) for board in boards]
        with self.gragh.as_default():
            probs, win_rate = self.session.run([tf.nn.softmax(self.y_conv), self.z_conv],
                feed_dict={self.training: False, self.x: features})
        vs1 = np.array(probs[:,0:go.N*go.N])
        add_p = config.policy_size-go.MAX_BOARD*go.MAX_BOARD
        vs2 = np.array(probs[:,-add_p:])
        vs = np.concatenate((vs1, vs2), axis=1)
        return vs.tolist(), win_rate*go.N*go.N

    def save_elo(self, fn):
        if fn is None:
            fn = self.save_file
            if fn is None:
                return
        with open(fn+'elo.txt', 'w') as f:
            f.writelines(str(self.elo)+'\n')
            f.writelines(str(self.game_num)+'\n')
            f.writelines(str(self.train_n)+'\n')
            f.writelines(str(self.loss_times)+'\n')
        
    def save(self, fn):
        if fn is None or not self.is_train:
            return
        path = os.path.dirname(fn)
        if not os.path.exists(path):
            os.makedirs(path)
        with self.gragh.as_default():
            self.saver.save(self.session, fn)
            self.save_leelaz_weights(path+"zigo.txt")
            self.save_elo(fn)

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
                print("Model saved in file: {}".format(save_path))
                leela_path = save_path + ".txt"
                self.save_leelaz_weights(leela_path)
                print("Leela weights saved to {}".format(leela_path))


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
        print("保存完毕。")
                
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
            #print("{}={}".format(i,len(ms)))
            if ms:
                for s in ms:
                    if s and s!="\n" and s!="\r\n":
                        w=float(s)
                        bf.write(struct.pack('f',w))

        #print(s)
        #print(w)
        txt.close()
        bf.close()
        print("保存完毕。")

