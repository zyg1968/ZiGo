#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

#import threading
#import time
#import queue
#import numpy as np
from tkinter import *
from PIL import Image
from qipan import *
import argparse
#import argh

from contextlib import contextmanager
#import os
#import random
#import sys

#from gtp import *
#from gtp_wrapper import *
#import network
import go
#import board
#from player import *
#import dataset
import config
#from intergtp import *
#import cgosplayer

@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f" % (message, (tock - tick)))

def gui():
    root=Tk()
    #设置窗口图标
    #root.iconbitmap('pic/zigo.ico')
    #width_px = root.winfo_screenwidth()
    #height_px = root.winfo_screenheight()
    #width_mm = root.winfo_screenmmwidth()
    #height_mm = root.winfo_screenmmheight()
    #width_in = width_mm / 25.4
    #height_in = height_mm / 25.4
    #width_dpi = width_px/width_in
    #height_dpi = height_px/height_in
    #root.geometry('%dx%d+%d+%d' % (width_px*0.95, height_px*0.9, 0, 0))
    root.resizable(width=True, height=True)
    root.title("智狗围棋")
    bjimg=Image.open(config.background)
    blackstone=Image.open(config.blackstone)
    whitestone=Image.open(config.whitestone)
    hqh=Image.open(config.blackbowls)
    bqh=Image.open(config.whitebowls)
    last=Image.open(config.lastmove)
    qp=QiPan('显示', root, bjimg, blackstone, whitestone, hqh, bqh, last)
    return qp, root

if __name__ == '__main__':
    config.read_cfg('config')
    config.logger.info("开始启动……")
    go.set_board_size(config.board_size)
    #network.start_time = time.time()
    parser = argparse.ArgumentParser()
    #subparsers = parser.add_subparsers(help='sub-command help')
    #gtpparser = subparsers.add_parser('gtp', help='用gtp协议下棋')
    #gtpparser.add_argument('engine', choices=['random','policy','randompolicy','mcts'], 
    #                       help='随机乱下/使用策略网络/使用策略网络随机下/使用蒙特卡洛搜索树算法')
    #gtpparser.add_argument('--read-file', help='训练好的网络数据文件，例如：--read-file=saved_models/20170718')
    #preparser = subparsers.add_parser('preprocess', help='预处理训练数据')
    #preparser.add_argument('path', help='训练用的棋谱文件 例如：data/kgs-*')
    #trainparser = subparsers.add_parser('train', help='训练')
    #trainparser.add_argument('path', help='预处理好的文件所在目录 例如：processed_data/')
    #trainparser.add_argument('--save-file', help='预处理好的文件所在目录 例如：--save-file=tmp/savedmodel')
    #trainparser.add_argument('--epochs', type=int, help='训练轮数 例如：--epochs=1')
    #trainparser.add_argument('--logdir', help='log文件 例如：--logdir=logs/my_training_run')
    qp, root=gui()
    #argh.add_commands(parser, [play, selftrain, preprocess, train])

    config.running = True
    #argh.dispatch(parser)
    root.mainloop()

