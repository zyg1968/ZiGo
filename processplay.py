#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

import time
import go
import strategies
import policy
import dataset
import utils
import os
import math
import random
import sgfparser
import copy
import threading
import shutil
import config
import numpy as np
import re
import sys
from scipy import stats
import multiprocessing as mp


def process_play(num=100, net=None, qp=None):
    if num<1:
        return
    #while config.running:
    print(time.strftime('%m-%d %H:%M:%S'), "：开始并行自战……")
    q = mp.Queue()
    thqueue = threading.Thread(target = receive_queue, args=(q,))
    thqueue.setDaemon(True)
    thqueue.start()
    nums = [num, num]
    pps = [mp.Process(name="SelfPlay"+str(i), target=thread_play, 
                        args=(q, net, n)) for i,n in enumerate(nums)]
    complete = 0
    for pp in pps:
        pp.daemon = True
        pp.start()
    thread_play(q, net, num, qp)
    for pp in pps:
        if config.running:
            pp.join()
        else:
            pp.terminate()
        #self.thread_play_random()
    print("自我对战生成训练数据完成。")

def receive_queue(q):
    while config.running:
        print(q.get(), file=sys.stderr)

def thread_play(queue, net=None, num=100, qp=None):
    config.read_cfg('config-b10f2')
    go.set_board_size(config.board_size)
    data = dataset.DataSet()
    name = mp.current_process().name
    for i in range(num):
        train_start = time.time()
        queue.put("{}--{:.0f}：开始下第{}盘棋……".format(name, time.time()-train_start, i+1))
        train_start = time.time()
        board = play(None, data, queue, net, qp, "sgf")
        queue.put("{}--{:.0f}：第{}盘棋下完。{}".format(name,time.time()-train_start, i+1, go.result_str(board.result)))
    if data.data_size>256:
        queue.put(name + "--保存训练数据……")
        data.save()
        queue.put(name + "--保存完毕！")
    

def thread_play_random(queue, num=1000, qp=None):
    config.read_cfg('config-b10f2')
    go.set_board_size(config.board_size)
    game_num = 0
    data = dataset.DataSet()
    name = mp.current_process().name
    config.running = True
    while config.running:
        train_start = time.time()
        board = strategies.simulate_game(board.Board(komi=7.5))
        queue.put("{}--{:.0f}：第{}盘随机盘面已准备就绪，共{}步，开始复盘……".format(name, time.time()-train_start, game_num, board.step))
        train_start = time.time()
        replay(board, data, queue, 100, qp)
        queue.put("{}--{:.0f}：第{}盘复盘完成。".format(name,time.time()-train_start, game_num))
        game_num += 1
        if game_num>=num:
            break
    if data.data_size>256:
        queue.put(name + "--保存训练数据……")
        data.save()
        queue.put(name + "--保存完毕！")

def replay(board, data, queue, times=80, qp=None):
    final_score = strategies.fast_score(board)
    mcts = strategies.RandomPlayer()
    rboard = board.copy()
    name = mp.current_process().name
    config.running = True
    for i in range(times):
        if not config.running:
            break
        if rboard.step<2:
            break
        rboard.undo()
        curboard = rboard.copy()
        if qp:
            qp.start(curboard)
        c=curboard.to_move
        while config.running and not curboard.is_gameover:
            move, move_values, points = mcts.suggest_move(curboard)
            curboard.win_rate = points if c == go.BLACK else -points
            data.add_from_node(curboard, move_values, points)
            if data.data_size%1024==5:
                queue.put("{}-- data size: {}".format(name, data.data_size))
            if data.data_size>12800:
                queue.put(name + "--保存训练数据……")
                data.save()
                data.clear()
                queue.put(name + "--保存完毕！")
            curboard.play_move(move)
            if qp:
                qp.update(curboard)

        score = curboard.win_rate if c==go.BLACK else -curboard.win_rate
        cscore = score-final_score
        cscore = cscore  if c == go.BLACK else -cscore
        if i%20==15:
            queue.put("{}--倒退{}步，全探查结果：{}，{}{}了{:.1f}子, {}步".format(name, i+1, 
                go.result_str(score), go.get_color_str(c), 
                "增加" if cscore>0 else "减少", abs(cscore), curboard.step))

def play(board=None, datas=None, queue=None, net=None, qp=None, sgfdir='sgf'):
    if not board:
        board = board.Board(komi=7.5)
    if qp:
        qp.start(board)
    passbw = 0
    mcts = strategies.MTCSSercher(net=net,qp=qp)
    caps = None
    while config.running:
        if board.step>go.N*go.N*2:
            board.result=0
            break
        c = board.to_move
        move, values, points = mcts.suggest_move(board)
        if board.step>go.N:
            datas.add_from_node(board, values, points)
        if datas.data_size%1024==5:
            queue.put("{}-- data size: {}".format(name, data.data_size))
        if datas.data_size>12800:
            print("保存训练数据……", end="", file=sys.stderr)
            datas.save()
            datas.clear()
            print("完毕！", file=sys.stderr)
        board.win_rate = points if c == go.BLACK else -points
        if (move is None or move == go.PASS):
            passbw |= (1 if c == go.BLACK else 2)
            if passbw == 3:
                board.play_move(go.PASS)
                score = strategies.fast_score(board)
                board.result = score / 2
                break
            else:
                board.play_move(go.PASS)
                continue
        elif move == go.RESIGN:
            board.play_move(move)
            board.result = go.N*go.N+1 if c==go.WHITE else -go.N*go.N-1
            '''msg = '%s方第%d手投子认负，对局结束, %s。' % (go.get_color_str(c),
                                           board.step, go.result_str(board.result))

            if qp:
                qp.show_message(msg=msg)'''
            break
        illegal, caps = board.play_move(move)
        if illegal == 0:
            passbw = 0
            if qp:
                qp.update(board)

    if sgfdir:
        dt = sgfdir + '/self_' + time.strftime('%Y-%m-%d_%H_%M_%S') + '.sgf'
        '''msg = '%.1f：\t保存sgf棋谱文件到%s' % (time.time() - policy.start_time, dt)
        if qp:
            qp.show_message(msg=msg)'''
        if not os.path.exists(sgfdir):
            os.makedirs(sgfdir)
        sgfparser.save_board(board, dt)
    return board

if __name__ == '__main__':
    process_play()
