import threading
import time
#import queue
import numpy as np
from tkinter import *
from PIL import Image
from qipan import *
import argparse
#import argh

from contextlib import contextmanager
import os
import random
import sys

from gtp import *
#from gtp_wrapper import *
import policy
import go
import board
from player import *
import selfplay
import dataset
import config
from intergtp import *

@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f" % (message, (tock - tick)))

def gui():
    root=Tk()
    def on_quit():
        config.running = False
        time.sleep(0.5)
        sys.stdout=None
        root.quit()
        root.destroy()
        exit()

    root.protocol("WM_DELETE_WINDOW", on_quit)
    #设置窗口图标
    #root.iconbitmap('pic/zigo.ico')
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()
    width_mm = root.winfo_screenmmwidth()
    height_mm = root.winfo_screenmmheight()
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    width_dpi = width_px/width_in
    height_dpi = height_px/height_in
    root.geometry('%dx%d+%d+%d' % (width_px*0.95, height_px*0.9, 0, 0))
    root.resizable(width=True, height=True)
    root.title("围棋")
    hpad = 30
    linespace = int((height_px - hpad) / (go.N+1.6))
    img=Image.open('pic/bj3.png')
    img=img.resize((int(height_px*0.95), int(height_px*0.95)), Image.ANTIALIAS)
    bjimg=ImageTk.PhotoImage(img)
    img=Image.open('pic/Black61.png')
    img=img.resize((int(linespace*0.95), int(linespace*0.95)), Image.ANTIALIAS)
    blackstone = ImageTk.PhotoImage(img)
    img=Image.open('pic/White61.png')
    img=img.resize((int(linespace*0.95), int(linespace*0.95)), Image.ANTIALIAS)
    whitestone = ImageTk.PhotoImage(img)
    img=Image.open('pic/hqh.png')
    img=img.resize((int(height_px*0.15), int(height_px*0.15)), Image.ANTIALIAS)
    hqh = ImageTk.PhotoImage(img)
    img=Image.open('pic/bqh.png')
    img=img.resize((int(height_px*0.15), int(height_px*0.15)), Image.ANTIALIAS)
    bqh = ImageTk.PhotoImage(img)
    img=Image.open('pic/last.png')
    img=img.resize((int(linespace*0.34), int(linespace*0.34)), Image.ANTIALIAS)
    last = ImageTk.PhotoImage(img)
    qp=QiPan('显示', root, bjimg, blackstone, whitestone, hqh, bqh, last, height_dpi, height_px)
    return qp, root

#play policy --read-file=saved_models/savedmodel
def play(player1='player', player2='ZiGo', gtpon=False, qp=None):
    thplay = PlayThread('下棋', qp, player1, player2, gtpon)
    thplay.setDaemon(True)
    thplay.start()

def preprocess(sgf_dir, save_dir="train_data"):
    if not sgf_dir or not os.path.isdir(sgf_dir):
        return
    path = '%s/f%d' % (save_dir, config.FEATURE_NUM)
    if not os.path.exists(path):
        os.makedirs(path)
    datas = dataset.DataSet()
    datas.save_dir = path
    for fn in os.listdir(datas.save_dir):
        filepath = os.path.join(datas.save_dir,fn)
        if os.path.isfile(filepath):
            datas.data_files.append(filepath)
    for fn in os.listdir(sgf_dir):
        print('从%s中生成训练数据……' % (fn), file=sys.stderr)
        filepath = os.path.join(sgf_dir,fn)
        if fn.endswith('.sgf'):
            datas.add_from_file(filepath)
            if datas.data_size>=3200:
                data.shuffle(1)
                datas.save()
                datas.clear()
    if datas.data_size>32:
        datas.save()
    print('%s中的棋谱生成训练数据完毕。保存到%s' % (sgf_dir, path))


def train(sgf_dir):
    if not sgf_dir or not os.path.isdir(sgf_dir):
        return
    datas = dataset.DataSet()
    if not os.path.exists(datas.save_dir):
        os.makedirs(datas.save_dir)
    for fn in os.listdir(datas.save_dir):
        filepath = os.path.join(datas.save_dir,fn)
        if os.path.isfile(filepath):
            datas.data_files.append(filepath)
    for fn in os.listdir(sgf_dir):
        print('从%s中生成训练数据……' % (fn))
        filepath = os.path.join(sgf_dir,fn)
        if fn.endswith('.sgf'):
            datas.add_from_file(filepath)
            if datas.data_size>=config.batch_size*4:
                datas.save()
                datas.clear()
    n = policy.PolicyNetwork(is_train = True, selftrain=False)
    datas.start_load(del_file = False)
    n.train(datas)
    print('保存模型和训练数据到'+config.save_dir)
    n.save_variables()
##    for fn in os.listdir(datas.save_dir):
##        filepath = os.path.join(datas.save_dir,fn)
##        if os.path.isfile(filepath):
##            os.remove(filepath)
    print('%s中的棋谱训练完毕。' % (sgf_dir))

class PlayThread (threading.Thread):
    def __init__(self, name, qp, player1, player2, gtpon=False):
        threading.Thread.__init__(self)
        self.name = name
        self.qp=qp
        self.running=False
        self.gtpon=gtpon
        self.gomanager=board.Board()
        self.gomanager.player1_name=player1
        self.gomanager.player2_name=player2
        self.net = None
        self.player1 = None
        self.player2 = None
        if player2 in policy_names or player1 in policy_names:
            self.net = policy.PolicyNetwork(is_train = False)
        if player1 in config.apps:
            cmd=config.apps[player1].split(",")
            self.player1=GTP_player(cmd, player1, go.BLACK)
            #self.player1=GTP_player(["d:/myprogram/ai/leela-zyg/msvc/x64/release/leelaz.exe", "-g", 
            #            "-w", "d:/myprogram/ai/leela-zyg/msvc/x64/release/config/zigo.txt"], player1)
        else:
            self.player1 = Player(player1, net=self.net, color=go.BLACK)     #执黑先行
        if player2 in config.apps:   
            cmd=config.apps[player2].split(",")
            self.player2=GTP_player(cmd, player2, go.WHITE)
        else:
            self.player2 = Player(player2, net=self.net, color=go.WHITE)   
        self.qp.start(self.gomanager)
        if gtpon:
            if self.player1.player_type == GTP:
                self.player1.boardsize(go.N)
                self.player1.komi(7.5)
            if self.player2.player_type == GTP:
                self.player2.boardsize(go.N)
                self.player2.komi(7.5)
            #self.engine=gtp.Engine(self)
            #if self.engine is None:
            #    sys.stderr.write("Gtp engine error!")
            #else:
            #    sys.stderr.write("GTP engine ready\n")
            #sys.stderr.flush()
            #it=InputThread('命令行下棋',self)
            #it.setDaemon(True)
            #it.start()

    def run(self):
        print ("开启线程： " + self.name, file=sys.stderr)
        self.running=True
        self.playloop()
        self.running=False
        if self.player1.player_type == GTP:
            self.player1.quit()
        if self.player2.player_type == GTP:
            self.player2.quit()
        print ("退出线程： " + self.name, file=sys.stderr)

    def playloop(self):
        caps = None
        while(self.running):
            c = self.gomanager.to_move
            player = self.player1 if c==self.player1.color else self.player2
            opponent = self.player2 if c==self.player1.color else self.player1

            if player.player_type == POLICY:
                coor, win_rate = player.get_move(self.gomanager, caps)
                self.gomanager.win_rate = win_rate
                #coor=self.gomanager.get_coor_from_vertex(vertex)  #从上倒下，左到右，0开始
            elif player.player_type == GTP:
                coorstr = player.genmove("b" if c==go.BLACK else "w")
                if coorstr.lower()=="resign":
                    coor=go.RESIGN
                else:
                    coor= go.get_coor_from_gtp(coorstr)
            else:
                self.qp.clicked=None
                while(not self.qp.clicked and not self.qp.havemoved):
                    time.sleep(0.1)
                if not self.qp.havemoved:
                    coor=self.qp.clicked
                    self.qp.clicked = None
            if coor is None or coor==go.PASS:
                if c == go.BLACK:
                    self.player1.passed += 1
                else:
                    self.player2.passed += 1
                print('%s pass了。' % (player.name), file=sys.stderr)
                if self.player1.passed>0 and self.player2.passed>0:
                    break
            if coor == go.RESIGN:
                self.gomanager.result = go.N*go.N+1 if c==go.WHITE else -(go.N*go.N+1)
                break
            if coor == (-3, 0):
                pstr = "undo"
                self.gomanager.undo()
                self.gomanager.undo()                
            else:
                pstr = go.get_cmd_from_coor(coor)
                #print('%s 准备走：%s(%d,%d)' % (player.name, pstr, coor[0], coor[1]))
                ill, caps = self.gomanager.play_move(coor, color=c)
                if ill>0:
                    print('%s方%s着法不合法，因为%s。' % (go.get_color_str(c), 
                        pstr, go.get_legal_str(ill)), file=sys.stderr)
                    continue
            if self.gtpon:
                gtp_reply = "无GTP Player"
                if pstr == "undo":
                    cmd = pstr
                    if opponent.player_type == GTP:
                        gtp_reply = opponent.send(cmd)
                else:
                    cmd = 'play %s %s' % ('black' if c==go.BLACK else 'white', pstr)
                #if player.player_type == GTP:
                #    gtp_reply = player.send(cmd)
                if opponent.player_type == GTP:
                    gtp_reply = opponent.send(cmd)
                #if gtp_reply=="":
                #    self.qp.show_message(msg='%s 走子成功：%s(%d,%d)' % (player.name, pstr, coor[0], coor[1]))
                    
            self.qp.update(self.gomanager)
            self.qp.havemoved = False
            self.running=config.running
            time.sleep(0.1)
        result=""
        if abs(self.gomanager.result) < go.N*go.N+1:
            if self.net:
                self.gomanager.result=strategies.final_score(self.gomanager, self.net)
                result = go.result_str(self.gomanager.result)
            elif self.player1.player_type == GTP:
                result = self.player1.score()
            elif self.player2.player_type == GTP:
                result = self.player2.score()
            else:
                self.gomanager.result = (self.gomanager.score())/2
                result = go.result_str(self.gomanager.result)
        else:
            result = go.result_str(self.gomanager.result)
        msg = "对局结束。%s" % (result)
        self.qp.show_message(msg=msg, status=msg)

    def set_size(self, n):
        go.set_board_size(n)
        self.gomanager.clear()

    def set_komi(self, komi):
        self.gomanager.komi = komi

    def clear(self):
        self.gomanager = board.Board(komi=7.5)

    def make_move(self, color, coords):
        ill, caps = self.gomanager.play_move(coords, color=color)
        if ill>0:
            return False
        return True

    def get_move(self, color):
        if self.gtppipe:
            self.gtppipe.send("genmove " + "b" if color==go.BLACK else "w")


#play --player1=mugo_policy --player2=mugo_policy --gtpon
#preprocess data/kgs-19-2014
#train processed_data --save-file=saved_models\savedmodel --epochs=1 --logdir=logs\train

if __name__ == '__main__':
    config.read_cfg('config')
    go.set_board_size(config.board_size)
    policy.start_time = time.time()
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

