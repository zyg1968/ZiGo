#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

import numpy as np
import go
import board
import gzip
import struct
import threading
import math
import os
import time
import config
import random
import sys
import pyautogui as pag
from PIL import Image, ImageTk

IMAGE_SIZE = 224

def select_random(pos, forbid_pass=True):
    possible_moves = pos.get_moves()
    if not possible_moves:
        return None
    move = random.choice(possible_moves)
    i = 0
    while forbid_pass and move == go.PASS:
        move = random.choice(possible_moves)
        i += 1
        if i>3:
            move = None
            break
    return move

def softmax(x):
    return (np.exp(x)/np.sum(np.exp(x),axis=0)).tolist()

def get_feature(board):
    return get_features(board, board.step)

def get_features(pos, step):
    history=int((config.FEATURE_NUM-config.PLAY_FEATURES)/config.STATE_FEATURES)
    onehot_features = np.zeros([config.FEATURE_NUM, go.N, go.N], dtype=np.int8)
    for i in range(history):
        c = pos.get_color(step-i)
        if len(pos.recent)<1 or pos.step == step-i:
            b = pos.stones
        elif step-i<1:
            b=pos.recent[0].stones
        elif step - i >= pos.step-1:
            b=pos.recent[-1].stones
        else:
            b = pos.get_stones(step-i)
        if config.STATE_FEATURES == 1:
            onehot_features[i] = b
        else:
            onehot_features[i*2,b==c] = 1
            onehot_features[i*2+1,b==-c] = 1
    c = pos.get_color(step)
    if config.PLAY_FEATURES == 2:
        if c==go.BLACK:
            onehot_features[config.FEATURE_NUM-2, :,:] = 1
        else:
            onehot_features[config.FEATURE_NUM-1,:,:] = 1
    elif config.PLAY_FEATURES == 1:
        onehot_features[config.FEATURE_NUM-1, :,:] = c
    return onehot_features

class DataSet():
    def __init__(self, qp, imgs=None, labels=None):
        self.qp=qp
        self.imgs = imgs if imgs else []
        self.labels = labels if labels else []
        self.data_size = len(self.labels)
        self.save_dir = config.data_dir
        self.index = 0
        self.isloading = False
        self.batch_size = config.batch_size
        self.data_files = []
        self.event = threading.Event()
        self.event.set()
        self.num_saved = 0
        self.loadth=None
        self.screen_width, self.screen_height = pag.size()

    def clear(self):
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        #self.states = np.ndarray([0,config.FEATURE_NUM,go.N,go.N], dtype=np.int)
        #self.move_values = np.ndarray([0,go.N*go.N+1], dtype=np.float32)
        #self.win_rates = np.ndarray([0], dtype=np.float32)
        self.imgs = []
        self.labels = []
        self.data_size = 0
        self.event.set()

    def get_save_num(self):
        for fn in os.listdir(self.save_dir):
            m = re.match(r"train([0-9]+)", fn)
            if m:
                num = m.group(1)
                num = int(num)
                return num
        return 0

    def shuffle(self, pct=1):
        k = int(self.data_size/self.batch_size*pct)*self.batch_size
        if k<self.batch_size:
            return
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        data = list(zip(self.imgs, self.labels))
        data1 = random.sample(data, k)
        self.imgs[:],self.labels[:]=zip(*data1)
        self.data_size=len(self.imgs)
        self.event.set()

    def get_batch(self, batch_size):
        if self.data_size<batch_size:
            return None,None
        s = 0
        e = batch_size
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        '''img = self.imgs[s:e]
        imgs = []
        for im in img:
            im = np.array(im).swapaxes(0, 2)
            im = Image.fromarray(im, mode='RGB')
            im.resize((224,224), Image.ANTIALIAS)
            imgs.append(np.array(im).swapaxes(0, 2).tolist())'''
        r = (np.array(self.imgs[s:e]), np.array(self.labels[s:e]))
        self.imgs = self.imgs[e:]
        self.labels = self.labels[e:]
        self.data_size = len(self.labels)
        self.event.set()
        return r

    def get_plane(self):
        im = pag.screenshot(region=(0, 0, 1800, 1800))
        im = im.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
        im = np.array(im).swapaxes(0, 2)
        return im

    def add_from_qp(self):
        while config.running:
            if self.data_size>6400:
                self.save()
                self.clear()
            self.qp.clear()
            left = random.randint(0,100)
            top = random.randint(0,160)
            width = random.randint(2400, 3200-left)
            height = random.randint(1200, 1800-top)
            self.qp.resize(left,top, width, height)
            self.qp.board = board.Board()
            #if random.randint(0,100)>50:
            #moves = random.choices(go.ALL_COORDS, k=random.randint(0,int(go.N*go.N*0.8)))
                #moves = [go.get_coor_from_vertex(move) for move in moves]
            for i in range(random.randint(0,int(go.N*go.N*0.8))):
                move = random.choice(go.ALL_COORDS)
                #move = select_random(self.qp.board, True)
                #self.qp.board.play_move(move)
                #self.qp.update()
                go.place_stones(self.qp.board.stones, random.randint(go.BLACK,go.WHITE), [move])
                self.qp.update()
                #time.sleep(0.1)
                im = pag.screenshot(region=(0, 0, 1800, 1800))
                im = im.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
                im = np.array(im).swapaxes(0, 2)
                while not self.event.is_set():
                    self.event.wait(0.1)
                self.event.clear()
                self.imgs.append(im)
                self.labels.append(self.qp.get_label())
                self.data_size = len(self.labels)
                self.event.set()
        if self.data_size>640:
            self.save()
            self.clear()

    def start_auto(self):
        if self.loadth and self.loadth.is_alive():
            return
        self.isloading=True
        self.event.set()
        #self.add_from_qp()
        self.loadth = threading.Thread(target = self.add_from_qp)
        self.loadth.setDaemon(True)
        self.loadth.start()

    def start_load(self, del_file=False):
        if self.loadth and self.loadth.is_alive():
            return
        self.isloading=True
        maxn = 0
        fns = os.listdir(self.save_dir)
        fn1 = max(fns, key=lambda x:int(x[5:]))
        maxn = int(fn1[5:])
        for fn in fns:
            n = int(fn[5:])
            if n>=maxn-50:
                filepath = os.path.join(self.save_dir,fn)
                if os.path.isfile(filepath):
                    self.data_files.append(filepath)
        self.event.set()
        random.shuffle(self.data_files)
        self.loadth = threading.Thread(target = self.load, args=(del_file,))
        self.loadth.setDaemon(True)
        self.loadth.start()

    def save(self, fn=None):
        if self.data_size<1:
            return
        if not fn:
            fn = self.save_dir+'/train'+str(self.num_saved)
            while os.path.isfile(fn):
                if fn not in self.data_files:
                    self.data_files.append(fn)
                self.num_saved += 1
                fn = self.save_dir+'/train'+str(self.num_saved)

        path = os.path.dirname(fn)
        if not os.path.exists(path):
            os.makedirs(path)
        config.logger.info('保存训练数据到%s中……' % (fn))
        batchs = math.ceil(self.data_size/self.batch_size)
        header_bytes = struct.pack('ii', self.data_size, self.batch_size)
        with gzip.open(fn, "wb", compresslevel=6) as f:
            f.write(header_bytes)
            for i in range(batchs):
                s=i*self.batch_size
                e=min((i+1)*self.batch_size, self.data_size)
                np.save(f, np.array(self.imgs[s:e]))
                np.save(f, np.array(self.labels[s:e]))
        self.num_saved += 1
        self.data_files.append(fn)
        config.logger.info('训练数据已保存到%s中，目前文件%d个' % (fn, len(self.data_files)))

    def load(self, del_file=True):
        self.isloading = True
        config.logger.info('读训练数据文件线程启动，共{}文件，持续喂数据中……'.format(len(self.data_files)))
        for fn in self.data_files:
            if not config.running or not self.isloading:
                break
            if not os.path.isfile(fn):
                continue
            while self.data_size>=config.batch_size*32:
                if not config.running or not self.isloading:
                    return
                time.sleep(1)
            config.logger.info('读取文件%s中的数据……' % (fn))
            with gzip.open(fn, "rb") as f:
                header_bytes = f.read(struct.calcsize('ii'))
                data_size, batch_size = struct.unpack('ii', header_bytes)
                batch_num = math.ceil(data_size/batch_size)
                for i in range(batch_num):
                    if not config.running or not self.isloading:
                        return
                    imgs = np.load(f) 
                    labels = np.load(f) 
                    while not self.event.is_set():
                        self.event.wait(0.1)
                    self.event.clear()
                    self.imgs += imgs.tolist()
                    self.labels += labels.tolist()
                    self.data_size = len(self.labels)
                    self.event.set()
                assert len(f.read()) == 0
                f.close()
            if del_file:
                os.remove(fn)
        self.isloading = False
        self.data_files = []
        config.logger.info('数据已全部读入，等待训练完成……')

