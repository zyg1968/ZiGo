#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

import threading,signal
import time
import numpy as np
from tkinter import *
from tkinter import messagebox
from scrolltext import ScrollText
from PIL import Image, ImageTk
import go
import board
import sys,os
import tkinter.filedialog
import sgfparser
import main
import config
import subprocess
import intergtp
#import dataset
import dbhash
import cgosplayer
import cgosmatchs
import player
import logging
from logging.handlers import RotatingFileHandler

def get_process_count(proname):
    p = os.popen('tasklist /FI "IMAGENAME eq %s"' % (proname))
    return p.read().count(proname)


class QiPan():
    def __init__(self, name, root, bj, bs, ws, hqh, bqh, last):
        # threading.Thread.__init__(self)
        # self.queue=msgqueue
        self.root = root
        self.img1 = bj
        self.img2 = bs
        self.img3 = ws
        self.img4 = hqh
        self.img5 = bqh
        self.img6 = last
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.height = self.screen_height
        self.board = None
        self.test_board = None
        self.clicked = None
        self.message = None
        self.havemoved = False
        self.showstep = False
        self.show_heatmap = False
        self.show_analyse = False
        self.heatmaps=[]
        self.heatmap_texts=[]
        self.var_name1 = StringVar()
        self.var_name2 = StringVar()
        self.player1 = None
        self.player2 = None
        self.current_stones = np.zeros(go.N2, dtype = np.int8)
        self.step = 0
        self.change_board = None
        self.change_step = False
        self.replay_step = None
        self.analyser = None
        self.pause_analyse = True
        self.canvas = None
        self.blackimg = None
        self.whiteimg = None
        self.hqh = None
        self.bqh = None
        self.cgoswin = None
        self.cgosplayer = None
        self.selfplay_connect = None
        self.initboard()
        self.resize(0,0,self.screen_width, self.screen_height)
        #self.resize(0,0,self.screen_width, self.screen_height)
        sys.stdout = self.gtpinfo
        sys.stderr = self.info
        fmt = '%(asctime)s [%(levelname)s]: %(message)s'
        datefmt = '%m-%d %H:%M:%S'
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(config.log_level)
        formatter = logging.Formatter(fmt, datefmt)
        console.setFormatter(formatter)
        config.logger.addHandler(console)
        #sys.stdin = self.cmd
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def on_resize(self,event):
        #self.width = event.width
        self.height = event.height
        # resize the canvas 
        minwh = event.height
        if abs(event.width-event.height)>10:
            minwh = min(event.width, event.height)
            self.canvas.config(width=minwh, height=minwh)
            return
        self.canvas_resize(event.x, event.y, minwh, minwh)

    def resize(self, left, top, width, height):
        self.root.geometry('%dx%d+%d+%d' % (width, height, left, top))

    def canvas_resize(self, left, top, width, height):
        if self.canvas:
            self.canvas.delete(ALL)
        self.current_stones=np.zeros(go.N2, dtype = np.int8)
        self.height = height
        #self.current_stones = np.zeros([go.N, go.N], dtype=np.int8)
        #self.board = None
        hpad = 0
        self.linespace = int((self.height-hpad) / (go.N+1))
        self.xs = int(self.linespace * 1)
        self.ys = int(self.linespace * 1)
        img=self.img1.resize((int(self.height), int(self.height)), Image.ANTIALIAS)
        self.bjimg=ImageTk.PhotoImage(img)
        img=self.img2.resize((int(self.linespace*0.95), int(self.linespace*0.95)), Image.ANTIALIAS)
        self.blackstone = ImageTk.PhotoImage(img)
        img=self.img3.resize((int(self.linespace*0.95), int(self.linespace*0.95)), Image.ANTIALIAS)
        self.whitestone = ImageTk.PhotoImage(img)
        img=self.img4.resize((int(self.height*0.15), int(self.height*0.15)), Image.ANTIALIAS)
        self.hqh = ImageTk.PhotoImage(img)
        if self.blackimg:
            self.blackimg.config(image = self.hqh)
        img=self.img5.resize((int(self.height*0.15), int(self.height*0.15)), Image.ANTIALIAS)
        self.bqh = ImageTk.PhotoImage(img)
        if self.whiteimg:
            self.whiteimg.config(image = self.bqh)
        img=self.img6.resize((int(self.linespace*0.34), int(self.linespace*0.34)), Image.ANTIALIAS)
        self.last = ImageTk.PhotoImage(img)
        self.drawboard()
        self.update()
        #self.canvas.addtag_all("all")

    def initboard(self):
        font = ("宋体", 10) #scalecanvas.Scale
        self.canvas = Canvas(self.root, bg='gray', width=self.height, height=self.height)
        self.stones = [None for i in range(go.N2)]
        self.steptexts = {}  #, log="/logs/info.log", log="/logs/important.log"
        self.info = ScrollText(self, self.root, width=75, font=font, padx=int(self.height * 0.004), pady=int(self.height * 0.002))
        self.gtpinfo = ScrollText(self, self.root, width=75, font=font, padx=int(self.height * 0.004), pady=int(self.height * 0.002))
        self.cmd = Entry(self.root, width=75, font=font)
        self.status = Label(self.root, text='盘面状态：', font=font, anchor=W)
        self.blackimg = Label(self.root, anchor=CENTER, image=self.hqh)
        self.player1_name = Label(self.root, width=16, text='岁的时候看', font=font, anchor=CENTER)
        self.player1_time = Label(self.root, width=16, text='1:11', font=font, anchor=CENTER)
        self.player1_eat = Label(self.root, width=16, text='eat1', font=font, anchor=CENTER)
        self.player1_winr = Label(self.root, width=16, text='胜率：', font=font, anchor=CENTER)
        self.whiteimg = Label(self.root, anchor=CENTER, image=self.bqh)
        self.player2_winr= Label(self.root, width=16, text='胜率：', font=font, anchor=CENTER)
        self.player2_eat = Label(self.root, width=16, text='eat2', font=font, anchor=CENTER)
        self.player2_time = Label(self.root, width=16, text='2:11', font=font, anchor=CENTER)
        self.player2_name = Label(self.root, width=16, text='pp', font=font, anchor=CENTER)
        cmdlabel = Label(self.root, width=16, text='命令行：', font=font, anchor=E)
        klb = Frame(self.root)#, height=int(self.height * 0.1), height=int(self.height * 0.1)
        klb1 = Frame(self.root)
        
        # grid
        padx = int(self.height * 0.004)
        pady = int(self.height * 0.002)
        padyb = self.height*0.006
        self.canvas.grid(row=0, column=0, rowspan=12, sticky=NSEW)
        self.blackimg.grid(row=0, column=1, stick=EW, padx=padx, pady=pady)
        self.player1_name.grid(row=1, column=1, stick=N, padx=padx, pady=pady)
        self.player1_time.grid(row=2, column=1, stick=N, padx=padx, pady=pady)
        self.player1_eat.grid(row=3, column=1, stick=N, padx=padx, pady=pady)
        self.player1_winr.grid(row=4, column=1, stick=N, padx=padx, pady=pady)
        klb.grid(row=5, column=1, stick=NS)
        klb1.grid(row=6, column=1, stick=NS)
        self.player2_winr.grid(row=7, column=1, stick=N, padx=padx, pady=pady)
        self.player2_eat.grid(row=8, column=1, stick=N, padx=padx, pady=pady)
        self.player2_time.grid(row=9, column=1, stick=N, padx=padx, pady=pady)
        self.player2_name.grid(row=10, column=1, stick=N, padx=padx, pady=pady)
        self.whiteimg.grid(row=11, column=1, stick=EW, padx=padx, pady=pady)
        self.info.grid(row=0, column=2, rowspan=6, sticky=NS)
        self.gtpinfo.grid(row=6, column=2, rowspan=6, sticky=NS)
        self.status.grid(row=12, column=0, sticky=EW, padx=padx, pady=padyb)
        cmdlabel.grid(row=12, column=1, sticky=EW, padx=padx, pady=padyb)
        self.cmd.grid(row=12, column=2, sticky=EW, padx=padx, pady=padyb)
        #self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(5, weight=1)
        self.root.rowconfigure(6, weight=1)
        #self.root.rowconfigure(4, weight=1)

        self.build_menu()
        self.scores = []
        self.last_img = None
        self.last_text = None
        self.root.bind('<Key>', self.on_key)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_undo)
        self.canvas.bind('<Control-Button-1>', self.on_ctl_click)
        self.canvas.bind("<Configure>", self.on_resize)

    def drawboard(self):
        font = ("宋体", 12)
        lw = int(self.linespace*0.01)
        dotr = int(self.linespace*0.1)
        self.canvas.create_image(int(self.bjimg.width() / 2.0),
                                 int(self.bjimg.height() / 2.0), image=self.bjimg)
        for i in range(go.N):
            self.canvas.create_text(i * self.linespace + self.xs, int(self.linespace*0.3),
                                    font=font, text=go.COLLUM_STR[i])
            self.canvas.create_text(i * self.linespace + self.xs, self.ys + int((go.N-0.3) * self.linespace),
                                    font=font, text=go.COLLUM_STR[i])
            self.canvas.create_line(i * self.linespace + self.xs, self.ys,
                                    i * self.linespace + self.xs, (go.N - 1) * self.linespace + self.ys, width=self.linespace*0.01)
            if i == 0 :
                self.canvas.create_line(self.xs-self.linespace*0.06 , self.ys - self.linespace*0.08 ,
                                        self.xs - self.linespace * 0.06,
                                        (go.N - 0.92) * self.linespace + self.ys, width=self.linespace*0.04)
            if i == go.N - 1:
                self.canvas.create_line((i+0.06) * self.linespace + self.xs, self.ys - self.linespace*0.08,
                                        (i + 0.06) * self.linespace + self.xs,
                                        (go.N - 0.92) * self.linespace + self.ys , width=self.linespace*0.04)
        for j in range(go.N):
            self.canvas.create_text(int(self.linespace*0.3), self.ys + j * self.linespace,
                                    font=font, text=str(go.N - j))
            self.canvas.create_text(self.xs + (go.N - 0.3) * self.linespace , self.ys + j * self.linespace,
                                    font=font, text=str(go.N - j))
            self.canvas.create_line(self.xs, j * self.linespace + self.ys,
                                    (go.N - 1) * self.linespace + self.xs, j * self.linespace + self.ys, width=self.linespace*0.01)
            if j == 0:
                self.canvas.create_line(self.xs -self.linespace*0.08, self.ys-self.linespace*0.06,
                                        (go.N - 0.92) * self.linespace + self.xs,
                                        self.ys - self.linespace * 0.06, width=self.linespace*0.04)
            if j == go.N - 1:
                self.canvas.create_line(self.xs - self.linespace * 0.08,
                                        (j+0.06) * self.linespace + self.ys,
                                        (go.N - 0.92) * self.linespace + self.xs ,
                                        (j+0.06) * self.linespace + self.ys, width=self.linespace*0.04)
        dots = list(map(lambda x: x * 6 + 3, range(int(go.N / 6))))
        for i in dots:
            for j in dots:
                self.canvas.create_oval(i * self.linespace + self.xs - dotr, j * self.linespace + self.ys - dotr,
                                        i * self.linespace + self.xs + dotr, j * self.linespace + self.ys + dotr,
                                        fill='black')

    def build_menu(self):
        # 在大窗口下定义一个菜单实例
        font = ("宋体", 12)
        menubar = Menu(self.root, font=font)
        # 在顶级菜单实例下创建子菜单实例, 去掉虚线tearoff=0
        self.filemenu = Menu(menubar, font=font, tearoff=0)
        # 为每个子菜单实例添加菜单项
        for lbl, cmd in zip(['打开', '保存'], [self.open, self.save]):
            self.filemenu.add_command(label=lbl, command=cmd)
        # 给子菜单添加分割线
        self.filemenu.add_separator()
        # 继续给子菜单添加菜单项
        for each in zip(['自动打谱', '手动打谱'],
                        [None, None]):
            self.filemenu.add_radiobutton(label=each, state=DISABLED)
        self.filemenu.add_separator()
        for lbl, cmd, acc in zip(['下一步', '上一步', '第一步', '最后一步'],
                                 [self.next_step, self.previous_step, self.first_step, self.last_step], 
                                 ['→', '←','↑','↓']):
            self.filemenu.add_radiobutton(label=lbl, command=cmd, accelerator=acc)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='退出', command=self.quit)

        self.funcmenu = Menu(menubar, font=font, tearoff=0)
        for lbl, cmd in zip(['对战', '继续对战'],#, '训练', '获取'
                            [self.start_game, self.continue_game]):#, self.get_traindata, self.train, self.get_board
            self.funcmenu.add_command(label=lbl, command=cmd)
        self.funcmenu.entryconfig(1, state=DISABLED)
        self.funcmenu.add_separator()
        self.funcmenu.add_command(label='查看CGOS对局', command=self.cgosview)
        self.funcmenu.add_command(label='开始CGOS挂机', command=self.cgosplay)
        self.funcmenu.add_separator()
        self.funcmenu.add_command(label='一键跑谱', command=self.train_play)

        self.actmenu = Menu(menubar, font=font, tearoff=0)
        self.chktest_play = tkinter.BooleanVar()
        for lbl, cmd, acc in zip(['形势判断', '关闭形势判断', '指点'],
                            [self.show_score, self.hidden_score, None],
                            ['E','F','']):
            self.actmenu.add_command(label=lbl, command=cmd, accelerator=acc)
        self.actmenu.add_checkbutton(label ='试下', 
            command = self.test_play, variable = self.chktest_play, accelerator='T')
        self.actmenu.add_separator()
        for lbl, cmd, acc in zip(['悔棋', '不走', '认输', '重新开始'],
                            [self.undo, self.move_pass, self.resign, self.restart],
                            ['U','P','R','S']):
            self.actmenu.add_command(label=lbl, command=cmd, accelerator=acc)
        self.actmenu.add_separator()
        self.actmenu.add_command(label='数子', command=self.show_result, accelerator='C')

        self.cfgmenu = Menu(menubar, font=font, tearoff=0)
        for lbl, cmd in zip(['棋盘设置', '棋手设置'],
                            [None, None]):
            self.cfgmenu.add_command(label=lbl, command=cmd)
        self.cfgmenu.add_separator()
        #cfgmenu.add_command(label="降低学习率", command = self.reduce_lr)
        self.chkstep = tkinter.BooleanVar()
        self.chkheatmap = tkinter.BooleanVar()
        self.chkanalyse = tkinter.BooleanVar()
        for lbl, v, cmd, acc in zip(['显示手数','显示热图','分析模式'],
                                    [self.chkstep,self.chkheatmap,self.chkanalyse],
                                    [self.change_show_step,self.change_show_heatmap,self.change_analyse],
                                    ['F2','F3','F4']):
            self.cfgmenu.add_checkbutton(label=lbl, variable=v, command=cmd, accelerator=acc)
        self.cfgmenu.add_separator()
        playermenu1 = Menu(self.cfgmenu, font=font, tearoff=0)
        playermenu2 = Menu(self.cfgmenu, font=font, tearoff=0)
        for each in ['人']:#, 'Policy', 'Ramdom', 'MTCS'
            playermenu1.add_radiobutton(label=each, variable=self.var_name1, value=each)
            playermenu2.add_radiobutton(label=each, variable=self.var_name2, value=each)
        playermenu1.add_separator()
        playermenu2.add_separator()
        for name, cmd in config.apps.items():
            playermenu1.add_radiobutton(label=name, variable=self.var_name1, value=name)
            playermenu2.add_radiobutton(label=name, variable=self.var_name2, value=name)

        self.var_name1.set("人")
        self.var_name2.set("zigo")
        self.cfgmenu.add_cascade(label='棋手1', menu=playermenu1)
        self.cfgmenu.add_cascade(label='棋手2', menu=playermenu2)
        self.cfgmenu.add_separator()
        #self.summary = tkinter.BooleanVar()
        #cfgmenu.add_checkbutton(label='打开观察表', variable=self.summary, command=self.change_summary)
        #self.summary.set(config.summary)

        self.helpmenu = Menu(menubar, font=font, tearoff=0)
        for lbl, cmd in zip(['帮助', '测试', '关于'],
                            [self.help, None, None]):
            self.helpmenu.add_command(label=lbl, command=cmd)
        # 为顶级菜单实例添加菜单，并级联相应的子菜单实例
        menubar.add_cascade(label='棋谱', menu=self.filemenu)
        menubar.add_cascade(label='功能', menu=self.funcmenu)
        menubar.add_cascade(label='操作', menu=self.actmenu)
        menubar.add_cascade(label='设置', menu=self.cfgmenu)
        menubar.add_cascade(label='帮助', menu=self.helpmenu)
        # 菜单实例应用到大窗口中
        self.root['menu'] = menubar

    def get_label(self):
        ml=int(go.N/2)
        return [go.N/go.MAX_BOARD, self.linespace/self.screen_height, 
                (ml * self.linespace + self.xs)/self.screen_height,
                (ml * self.linespace + self.ys)/self.screen_height]

    def change_show_step(self):
        self.showstep = self.chkstep.get()
        self.update()

    def change_show_heatmap(self):
        self.show_heatmap = self.chkheatmap.get()
        self.update_heatmap()

    def change_analyse(self):
        if self.player1 or self.player2:
            messagebox.showerror("不能分析", "你在下棋哦，别想作弊啊！")
            return
        self.show_analyse = self.chkanalyse.get()
        if self.show_analyse:
            self.show_heatmap = True
            self.chkheatmap.set(True)
            cmd =config.apps["zigo"].split(",")
            self.analyser = intergtp.GTPPlayer(cmd, "zigo", go.WHITE, True)
            #self.analyser.stop_analyse()
            if self.board and self.board.step>0:
                #for i in range(self.step):
                #    pm = self.board.recent[i]
                #    vertex = go.get_cmd_from_coor(pm.move)
                #    self.analyser.play(vertex, pm.color)
                sgf = sgfparser.SgfParser(board=self.board.get_board(self.step))
                sgf.save("temp.sgf")
                path = os.path.realpath('temp.sgf')
                self.analyser.loadsgf(path, self.step)
                time.sleep(0.6)
            self.analyser.analyse()
        elif self.analyser:
            self.analyser.stop_analyse()
            time.sleep(0.6)
            self.analyser.quit()
            self.analyser = None
            self.show_heatmap = False
            self.chkheatmap.set(self.show_heatmap)
            #if os.path.exists("temp.sgf"):
            #    os.remove("temp.sgf")

    def analyse(self):
        self.pause_analyse = not self.pause_analyse
        if self.analyser:
            self.analyser.send("pause_analyse")

    def start(self, pos=None, p1name=None, p2name=None):
        if pos is None:
            self.board = board.Board()
        else:
            self.clear()
            self.board = pos
            if not p1name:
                p1name = pos.player1_name
            if not p2name:
                p2name = pos.player2_name
            self.player1_name['text'] = p1name
            self.player2_name['text'] = p2name
        #pos = self.board.get_board(self.step)
        self.update()

    def update(self, pos=None):
        if pos is None:
            pos = self.board
            if not pos:
                return
            self.step = pos.step
        elif self.step>pos.step:
            self.step = pos.step

        if self.step<=0 and (not self.board or self.board.step==0):
            self.funcmenu.entryconfig(1, state=DISABLED)
        changed = np.where(pos.stones!= self.current_stones)
        for coor in changed[0]:
            if coor == go.PASS or coor == go.RESIGN:
                continue
            if self.current_stones[coor] != go.EMPTY:
                self.pickup(coor)
            else:
                self.downstone(pos.stones[coor], coor)
        self.current_stones = pos.stones.copy()

        if self.step < 1:
            if self.last_text:
                self.canvas.delete(self.last_text)
            if self.last_img:
                self.canvas.delete(self.last_img)
            return
        move = pos.recent[self.step - 1].move
        if move == go.PASS or move == go.RESIGN:
            return
        c = pos.recent[self.step - 1].color
        font = ("黑体", int(12* self.linespace/80))
        x, y = go.unflatten_coords(move)
        if self.showstep:
            if not self.steptexts:
                if self.last_text:
                    self.canvas.delete(self.last_text)
                    self.last_text = None
                self.steptexts = {}
                for i,pm in enumerate(pos.recent):
                    x1,y1=go.unflatten_coords(pm.move)
                    self.steptexts[pm.move] = (self.canvas.create_text(x1 * self.linespace + self.xs,
                                        y1 * self.linespace + self.ys, text=str(i+1),
                                        font=font, fill='white' if pm.color == go.BLACK else 'black'))
            else:
                for coor in changed:
                    if coor == go.PASS or coor == go.RESIGN:
                        continue
                    step = pos.get_step_from_move(coor)
                    if self.current_stones[coor] == go.EMPTY:
                        self.canvas.delete(self.steptexts[coor])
                    else:
                        color = pos.get_color(step)
                        x1, y1 = go.unflatten_coords(coor)
                        self.steptexts[coor] = (self.canvas.create_text(x1 * self.linespace + self.xs,
                                                y1 * self.linespace + self.ys, text=str(step+1),
                                                font=font, fill='white' if color == go.BLACK else 'black'))
        else:
            if self.steptexts:
                for m,t in self.steptexts.items():
                    self.canvas.delete(t)
                self.steptexts.clear()
            if self.last_text:
                self.canvas.delete(self.last_text)
            self.last_text = self.canvas.create_text(x * self.linespace + self.xs,
                                                     y * self.linespace + self.ys, text=str(self.step),
                                                     font=font, fill='white' if c == 1 else 'black')

        if self.last_img:
            self.canvas.delete(self.last_img)
        self.last_img = self.canvas.create_image((x + 0.3) * self.linespace + self.xs,
                                                 (y + 0.3) * self.linespace + self.ys, image=self.last)

        #score = pos.score()
        self.player1_eat['text'] = '吃子：%d' % (pos.caps[0])
        self.player2_eat['text'] = '吃子：%d' % (pos.caps[1])
        self.update_winrate()

    def update_winrate(self):
        values = None
        color = go.WHITE
        if self.analyser:
            values = self.analyser.values
        elif self.player1 and self.player1.player_type == player.GTP:
            color = go.BLACK
            values = self.player1.values
        elif self.player2 and self.player2.player_type == player.GTP:
            color = go.WHITE
            values = self.player2.values
        elif self.cgosplayer and self.cgosplayer.is_play and \
            self.cgosplayer.player.player_type == player.GTP:
            color = self.cgosplayer.player.color
            values = self.cgosplayer.player.values
        if values:
            m = list(values.keys())[0]
            wb = values[m].winrate
            if color == go.WHITE:
                self.player1_winr["text"] = "白：黑胜率{:.1f}".format(100-wb)
                self.player2_winr["text"] = "白：白胜率{:.1f}".format(wb)
            else:
                self.player1_winr["text"] = "黑：黑胜率{:.1f}".format(wb)
                self.player2_winr["text"] = "黑：白胜率{:.1f}".format(100-wb)
        #if self.analyser and self.info.value_changed:
        #    self.update_heatmap(self.analyser.values)
        #    self.info.value_changed = False
            
        #self.status['text'] = '{}手，估计黑方胜率：{:.1f}，白方胜率：{:.1f}，{}领先{:.1f}子。'.format(pos.step,\
        #                                                       wb, ww, go.get_color_str(score), abs(score))
        #self.info.see(END)

    def update_heatmap(self, values=None):
        if self.heatmaps:
            for hm in self.heatmaps:
                self.canvas.delete(hm)
            self.heatmaps = []
        if self.heatmap_texts:
            for ht in self.heatmap_texts:
                self.canvas.delete(ht)
            self.heatmap_texts=[]
        if self.analyser is None:
            return
        if self.show_heatmap:
            if values is None:
                values = self.analyser.values
            if values is None:
                return
            #vs={}
            font = ("宋体", int(8* self.linespace/80))
            r = self.linespace*0.5
            #for m,v in values.items():
                #m = utils.unflatten_coords(i)
                #if self.current_stones[m] == go.EMPTY and v>0:
                #    vs[m] = values[i]
            allvisits=0
            for m,v in values.items():
                allvisits += v.visits
            j = 0
            for m,v in values.items():
                j += 1
                if j>10:
                    break
                vm = int((v.visits/allvisits)*164)
                red = 255
                g = min(164, max(0, 164-vm))
                b = min(164, max(0, 164-vm))
                color ='#%02X%02X%02X' %(red, g, b) 
                x,y=go.unflatten_coords(m)
                hm = self.canvas.create_oval(x * self.linespace + self.xs - r, y * self.linespace + self.ys - r,
                                        x * self.linespace + self.xs + r, y * self.linespace + self.ys + r,
                                        fill=color)
                self.heatmaps.append(hm)
                ht = self.canvas.create_text(x * self.linespace + self.xs,
                                y * self.linespace + self.ys - int(r/3), text=str("{:.1f}".format(v.winrate)),
                                font=font, fill='black')
                self.heatmap_texts.append(ht)
                ht2 = self.canvas.create_text(x * self.linespace + self.xs,
                                y * self.linespace + self.ys+int(r/3), text=str(v.visits),
                                font=font, fill='black')
                self.heatmap_texts.append(ht2)


    def clear(self):
        self.canvas.delete(ALL)
        self.drawboard()
        self.current_stones = np.zeros(go.N2, dtype = np.int8)
        self.board = None

    def show(self, pos, next_move=None):
        self.clear()
        self.showstep = True
        for i, pm in enumerate(pos.recent):
            self.downstone(pm.color, pm.move)
            if pm.captured:
                for p in pm.captured:
                    self.pickup(p)
        if next_move:
            self.downstone(pos.to_move, next_move)
    
    def on_click(self, event):
        x, y = self.coortoline(event.x, event.y)
        if x < 0 or x > go.N - 1 or y < 0 or y > go.N - 1:
            return
        self.clicked = y*go.N+x
        clicked = self.clicked
        if not self.board:
            self.board=board.Board()
        c = self.board.to_move
        self.board.play_move(clicked)
        self.update()
        if self.show_analyse and self.analyser:
            vertext =go.get_cmd_from_coor(clicked)
            self.analyser.play(vertext, c)

    def on_undo(self, event):
        self.undo()

    def downstone(self, c, move):
        if move<0 or move >= go.N2:
            return
        x, y = move % go.N, move//go.N
        tag = go.get_cmd_from_coor(move)
        self.stones[move] = self.canvas.create_image(x * self.linespace + self.xs, y * self.linespace + self.ys,
                                                     image=self.blackstone if c == go.BLACK else self.whitestone,
                                                     tags=(tag))
        '''
        if self.last_move:
            self.canvas.delete(self.last_move)
        self.last_move = self.canvas.create_rectangle(i*self.linespace+self.xs-lwh,
            j*self.linespace+self.ys-lwh, i*self.linespace+self.xs+lwh,
            j*self.linespace+self.ys+lwh, fill='white' if c==1 else 'black')
        '''

    def pickup(self, coor):
        tag = go.get_cmd_from_coor(coor)
        if self.stones[coor]:
            self.canvas.delete(self.stones[coor])

    def show_message(self, msg=None, status=None):
        if msg:
            # if self.message:
            #    self.canvas.delete(self.message)
            # self.message = self.canvas.create_text(360, 650, text=msg)
            #txt = self.info.get(0.0, END)
            #if txt.count('\n') > 200:
            #    self.info.delete(0.0, 20.0)
            self.info.insert(END, msg + '\n')
            self.info.see(END)
        if status:
            self.status['text'] = status

    def hidden_score(self):
        if len(self.scores) > 0:
            for rect in self.scores:
                self.canvas.delete(rect)

    def show_score(self, pos=None):
        if pos is None:
            pos = self.board
            if not pos:
                return 0
        changed = np.where(pos.stones != self.board.stones)
        add_black = []
        add_white = []
        for x in changed[0]:
            if pos.stones[x] == go.BLACK:
                add_black.append(x)
            elif pos.stones[x] == go.WHITE:
                add_white.append(x)
        self.hidden_score()
        lwh = int(self.height * 0.01)
        score,bs,ws,has_uk = pos.scorebw()
        bs.extend(add_black)
        ws.extend(add_white)
        self.show_message(status="TT规则判断：{}".format(go.get_point_str(score)))
        for b in bs:
            x,y=go.unflatten_coords(b)
            self.scores.append(self.canvas.create_rectangle(x * self.linespace + self.xs - lwh,
                                                            y * self.linespace + self.ys - lwh,
                                                            x * self.linespace + self.xs + lwh,
                                                            y * self.linespace + self.ys + lwh, fill='black'))
        for b in ws:
            x,y=go.unflatten_coords(b)
            self.scores.append(self.canvas.create_rectangle(x * self.linespace + self.xs - lwh,
                                                            y * self.linespace + self.ys - lwh,
                                                            x * self.linespace + self.xs + lwh,
                                                            y * self.linespace + self.ys + lwh, fill='white'))
        return score

    def clear_dead(self):
        cp = None
        if self.player1 and self.player1.player_type == player.GTP:
            cp = self.player1
        elif self.player2 and self.player2.player_type == player.GTP:
            cp = self.player2
        elif self.analyser:
            cp = self.analyser
        if cp is None:
            return go.delete_dead(self.board)
        cdstones = np.array(cp.clear_dead())
        pos = self.board.copy()
        pos.stones = cdstones
        return pos

    def show_result(self):
        #pos = go.simulate_game(self.board)
        pos = self.clear_dead()
        #self.update(pos)
        self.board.points = self.show_score(pos)
        self.board.result = self.board.get_result()
        result = go.result_str(self.board.result)
        self.show_message(msg=result, status = result)

    def coortoline(self, x, y):
        lx = int((x - (self.xs - self.linespace / 2.0)) / self.linespace)
        ly = int((y - (self.ys - self.linespace / 2.0)) / self.linespace)
        return lx, ly

    def play(self, gtpmove, color=None):
        coord = go.get_coor_from_gtp(gtpmove)
        self.play_move(coord)
    
    def undo(self):
        if self.show_analyse and self.analyser:
            self.analyser.send("undo")
            self.board.undo()
            self.analyser.send("undo")
            self.board.undo()
            self.update()
        elif (self.player1 and self.player1.player_type==player.PLAYER) or \
            (self.player2 and self.player2.player_type==player.PLAYER):
            self.clicked=go.UNDO
        #self.board.undo()
        #self.update()

    def komi(self, komi):
        if not self.board:
            self.board = board.Board(komi=komi)
        else:
            self.board.komi = komi

    def boardsize(self, size):
        go.set_board_size(size)
        self.clear_board()

    def clear_board(self):
        self.clear()
        self.board = board.Board()

    def score(self):
        return self.board.score()

    def handicap(self, handicap, handicap_type="fixed"):
        self.board = board.Board()
        self.board.set_handicap(handicap)

    def set_time(self, time_settings):
        if not time_settings:
            return
        self.time = time_settings
        ts = time_settings.split(' ')
        t = 0
        cd = 0
        ct = 0
        m = 0
        s=0
        if len(ts)>0:
            t = int(ts[0])
            m = int(t/60)
            s = t%60
        if len(ts)>1:
            cd = int(ts[1])
        if len(ts)>2:
            ct = int(ts[2])
        self.player1_time['text'] = '{}:{} {} {}'.format(m,s, cd, ct)  
        self.player2_time['text'] = '{}:{} {} {}'.format(m,s, cd, ct)  
        if self.player1:
            self.player1.set_time(time_settings)
            #if self.player1.player_type == player.GTP:
            #    self.player1.set_time(time_settings)
        if self.player2:
            self.player2.set_time(time_settings)
            #if self.player2.player_type == player.GTP:
            #    self.player2.set_time(time_settings)

    def time_left(self, color, msec):
        sec = int(msec)
        m = int((sec)/60)
        s = sec % 60
        cd = 0
        ct = 0
        if color == go.BLACK:
            if self.player1:
                cd = self.player1.countdown or 0
                ct = self.player1.counttimes or 0
            self.player1_time['text'] = '{}:{} {} {}'.format(m,s, cd, ct)  
        else:
            if self.player2:
                cd = self.player2.countdown or 0
                ct = self.player2.counttimes or 0
            self.player2_time['text'] = '{}:{} {} {}'.format(m,s, cd, ct)  

    def play_move(self, coord, color=None):
        if self.board is None:
            self.board = board.Board()
        if coord == go.UNDO:
            return self.undo()
        self.board.play_move(coord, color)
        self.update()

    # 菜单命令
    def quit(self):
        config.running = False
        if self.analyser:
            self.analyser.stop_analyse()
            self.analyser.quit()
            self.analyser=None
        if self.player1 and not self.player1.is_quit and self.player1.player_type == player.GTP:
            self.player1.quit()
            self.player1=None
        if self.player2 and not self.player2.is_quit and self.player2.player_type == player.GTP:
            self.player2.quit()
            self.player2=None
        if self.selfplay_connect:
            self.selfplay_connect.write('q'+chr(13))
            self.selfplay_connect = None
        #if self.info.log:
        #    self.info.save_log()
        #if self.gtpinfo.log:
        #    self.gtpinfo.save_log()
        #time.sleep(0.5)
        sys.stdout = None
        sys.stderr = None
        self.root.quit()
        self.root.destroy()
        exit()

    def on_key(self, event):
        if event.widget == self.cmd:
            if event.keysym == 'Return':
                cmd = self.cmd.get()
                if self.analyser:
                    self.analyser.send(cmd)
                if self.player1 and self.player1.player_type == player.GTP:
                    self.player1.send(cmd)
                if self.player2 and self.player2.player_type == player.GTP:
                    self.player2.send(cmd)
            return
        if event.keysym == 'Next' or event.keysym == 'Right':
            self.next_step()
        elif event.keysym == 'Prior' or event.keysym == 'Left':
            self.previous_step()
        elif event.keysym == 'Up' or event.keysym == 'Home':
            self.first_step()
        elif event.keysym == 'Down' or event.keysym == 'End':
            self.last_step()
        elif event.keysym == 'u' or event.keysym == 'U':
            self.undo()
        elif event.keysym == 'p' or event.keysym == 'P':
            self.move_pass()
        elif event.keysym == 'r' or event.keysym == 'R':
            self.resign()
        elif event.keysym == 's' or event.keysym == 'S':
            self.start()
        elif event.keysym == 'c' or event.keysym == 'C':
            self.show_result()
        elif event.keysym == 'e' or event.keysym == 'E':
            self.show_score()
        elif event.keysym == 'f' or event.keysym == 'F':
            self.hidden_score()
        elif event.keysym == 't' or event.keysym == 'T':
            self.chktest_play()
        elif event.keysym == 'F2':
            self.chkstep.set(not self.showstep)
            self.change_show_step()
        elif event.keysym == 'F3':
            self.chkheatmap.set(not self.show_heatmap)
            self.change_show_heatmap()
        elif event.keysym == 'F4':
            self.chkanalyse.set(not self.show_analyse)
            self.change_analyse()
        elif event.keysym == 'space':
            self.analyse()
        elif event.keysym == 'Escape':
            if self.change_board:
                self.board=self.change_board
                self.showstep = self.change_step
                self.show_heatmap = True
                self.update_heatmap()
                self.change_board = None
                pos = self.board.get_board(self.replay_step)
                self.update()

    def on_ctl_click(self, event):
        if not self.analyser:
            return
        x, y = self.coortoline(event.x, event.y)
        if y < 0 or y > go.N - 1 or x < 0 or x > go.N - 1:
            return
        p = go.flatten_coords((x,y))
        if p in self.analyser.values.keys():
            pos = self.board.copy()
            self.change_step = self.showstep
            self.replay_step = self.step
            self.showstep = False
            self.show_heatmap = False
            self.update_heatmap(None)
            self.board = self.board.get_board(self.step)
            self.board.step = 0
            self.board.recent = []
            self.update(self.board)
            self.showstep=True
            for i, move in enumerate(self.analyser.values[p].nextmoves.split(" ")):
                if i>config.max_changes:
                    break
                m = go.get_coor_from_gtp(move)
                if m==go.PASS or m==go.RESIGN or m is None:
                    continue
                self.board.play_move(m)
            self.update()
            self.change_board = pos

    def open(self):
        fn = tkinter.filedialog.askopenfilename(filetypes=[("sgf格式", "sgf")])
        pos = sgfparser.get_sgf_board(fn)
        self.start(pos)
        self.funcmenu.entryconfig(1, state=ACTIVE)
        if self.analyser:
            #path = os.path.realpath(fn)
            self.analyser.loadsgf(fn, self.board.step)


    def save(self):
        fn = tkinter.filedialog.asksaveasfilename(filetypes=[("sgf格式", "sgf")])
        sgf = sgfparser.SgfParser(board=self.board)
        if not fn.endswith(".sgf"):
            fn += ".sgf"
        sgf.save(fn)
        
    def first_step(self):
        self.step = 0
        if self.analyser:
            self.analyser.send("clear_board")
        pos = self.board.get_board(self.step)
        self.update(pos)

    def last_step(self):
        if self.analyser:
            for i in range(self.board.step-self.step):
                pm = self.board.recent[self.step+i]
                self.analyser.play(go.get_cmd_from_coor(pm.move), pm.color)
        self.step = self.board.step
        self.update()

    def next_step(self):
        if self.step >= self.board.step:
            return
        if self.analyser:
            c = self.board.get_color(self.step)
            move = self.board.recent[self.step].move
            vertex = go.get_cmd_from_coor(move)
            self.analyser.play(vertex, c)
        self.step += 1
        pos = self.board.get_board(self.step)
        self.update(pos)

    def previous_step(self):
        if self.step < 1:
            return
        if self.analyser:
            self.analyser.send("undo")
        self.step -= 1
        pos = self.board.get_board(self.step)
        self.update(pos)

    def move_pass(self):
        self.clicked = go.PASS
        c = self.board.to_move
        self.board.play_move(self.clicked)
        self.update()
        if self.show_analyse and self.analyser:
            vertext =go.get_cmd_from_coor(self.clicked)
            self.analyser.play(vertext, c)

    def resign(self):
        self.clicked = go.RESIGN

    def test_play(self):
        test_play = self.chktest_play.get()
        if test_play:
            self.test_board = self.board.copy()
            self.board = self.board.get_board(self.step)
        else:
            self.board = self.test_board
            self.test_board = None
            self.update()

    def start_game(self):
        if self.analyser:
            messagebox.showerror("不能对战", "你正在分析啊，别想作弊哦！")
        self.board = board.Board()
        p1 = self.var_name1.get().lower()
        p2 = self.var_name2.get().lower()
        if self.player1 is None or self.player1.name!=p1 or self.player1.is_quit:
            if p1 in config.apps:
                cmd=config.apps[p1].split(",")
                self.player1= intergtp.GTPPlayer(cmd, p1, go.BLACK, log='gtp1')
            elif p1=='人':
                self.player1 = player.Player(p1, color=go.BLACK)     #执黑先行
        if self.player2 is None or self.player2.name!=p2 or self.player2.is_quit:
            if p2 in config.apps:   
                cmd=config.apps[p2].split(",")
                try:
                    self.player2=intergtp.GTPPlayer(cmd, p2, go.WHITE, log='gtp2')
                except:
                    config.logger.exception(cmd)
                    self.quit()
            elif p2=='人':
                self.player2 = player.Player(p2, color=go.WHITE)   
        self.show_message(status='{}和{}开始对战……'.format(self.player1.name, self.player2.name))
        player.play(player1=self.player1, player2=self.player2, qp=self)

    def continue_game(self):
        if self.step<=0:
            self.board=board.Board()
        if self.board.step!=self.step:
            self.board = self.board.get_board(self.step)
        sgf = sgfparser.SgfParser(board=self.board.get_board(self.step))
        sgf.save("temp.sgf")
        p1 = self.var_name1.get().lower()
        p2 = self.var_name2.get().lower()
        if self.player1 is None:
            if p1 in config.apps:
                cmd=config.apps[p1].split(",")
                self.player1= intergtp.GTPPlayer(cmd, p1, go.BLACK, log='gtp1')
                self.player1.loadsgf(os.path.curdir + "/temp.sgf", self.step)
            elif p1=='人':
                self.player1 = player.Player(p1, color=go.BLACK)     #执黑先行
        if self.player2 is None:
            if p2 in config.apps:   
                cmd=config.apps[p2].split(",")
                self.player2=intergtp.GTPPlayer(cmd, p2, go.WHITE, log='gtp2')
                self.player2.loadsgf(os.path.curdir + "/temp.sgf", self.step)
            elif p2=='人':
                self.player2 = player.Player(p2, color=go.WHITE)   
            player.play(player1=self.player1, player2=self.player2, qp=self, start_color = self.board.to_move)

    def cgosview(self):
        self.cgoswin = cgosmatchs.CGOSWindow(self.height)
        if self.cgosplayer is None:
            self.cgosplayer = cgosplayer.CGOSConnector(self, None)
        #self.cgoswin.cgosplayer=self.cgosplayer
        cgosth = threading.Thread(target = cgosplayer.start_cgos, args=(self.cgosplayer,))
        cgosth.setDaemon(True)
        cgosth.start()
        self.cgoswin.top.mainloop()

    def cgosplay(self):
        if not config.username and not config.password:
            messagebox.showinfo('提示', '请先在CGOS申请账号，在config.ini中设置好账号、密码和engine_name才能在CGOS下棋。')
            return
        pn = self.var_name2.get()
        if pn not in config.apps.keys():
            messagebox.showinfo('提示', '请在“设置”菜单“棋手2”处选择一个软件作为CGOS的下棋引擎。\n（不能选人哈！）')
            return
        cmd=config.apps[pn].split(",")
        player = intergtp.GTPPlayer(cmd, pn)
        self.cgosplayer = cgosplayer.CGOSConnector(self, player)
        cgosth = threading.Thread(target = cgosplayer.start_cgos, args=(self.cgosplayer,))
        cgosth.setDaemon(True)
        cgosth.start()

    def train_play(self):
        #os.chdir('dist/leelaz')   # 将当前工作目录改变为``
        bf = None
        for fn in os.listdir(config.lz_dir):
            if re.match("storefile.+\.bin", fn):
                bf = fn
                break
        if bf is not None:
            with open(os.path.join(config.lz_dir, bf), 'r') as f:
                for line in f.readlines():
                    m = re.match('sgf ([a-zA-Z0-9]+)\n', line)
                    if m:
                        sgf = os.path.join(config.lz_dir, m.group(1)+".sgf")
                        pos = sgfparser.get_sgf_board(sgf)
                        self.start(pos)
                        break

        self.selfplay_connect = intergtp.GTP_connection([config.lz_dir+'/autogtp.exe',], connect_type=intergtp.SELF_PLAY)
        self.selfplay_connect.setDaemon(True)
        self.selfplay_connect.start()
        #os.chdir('dist/leelaz')

    def update_selfplay(self):
        if self.selfplay_connect and self.selfplay_connect.moves:
            for move in self.selfplay_connect.moves:
                self.play_move(move)
      
    '''def train(self):
        self.show_message('开始训练……')
        datas = dataset.DataSet(self)
        datas.start_load()
        self.train_net = network.Network(self.screen_height, is_train=True)
        self.show_message(status='正在训练……')
        trainth = threading.Thread(target = self.train_net.train, args=(datas,))
        trainth.setDaemon(True)
        trainth.start()

    def get_board(self):
        #path = tkinter.filedialog.askdirectory()
        self.show_message('开始获取棋谱%s中的数据……')
        datas = dataset.DataSet(self)
        plane = datas.get_plane()
        #print(plane.shape())
        #self.get_label()
        #self.train_net = network.Network(self.screen_height, is_train=False)
        #vs = self.train_net.run(plane)
        #print(vs)
        #testoval = self.canvas.create_oval(vs[2]-20, vs[3]-20,vs[2]+20,vs[3]+20)
        print("board size:{}, line space:{}, x:{}, y:{}".format(vs[0], vs[1], vs[2], vs[3]))
    '''

    def reduce_lr(self):
        self.self_train.reduce_lr()

    def restart(self):
        if self.analyser:
            self.analyser.send("clear_board")
        self.start()
    
    '''
    def get_traindata(self):
        #path = tkinter.filedialog.askdirectory()
        self.show_message('开始自我训练……')
        datas = dataset.DataSet(self)
        datas.start_auto()
        #self.train_net = network.Network(self.screen_height, is_train=True)
        self.show_message(status='正在进行增强学习……')
        #trainth = threading.Thread(target = self.train_net.train, args=(datas,))
        #trainth.setDaemon(True)
        #trainth.start()

    def cfg_window(self):
        cfg = config.ConfigWindow(self.height*1, self.height*0.6)
        cfg.mainloop()
    '''

    def help(self):
        #tkMessageBox.showinfo(title='aaa', message='bbb')
        helpwin = HelpWindow(self.height)
        helpwin.mainloop()

    '''def change_summary(self):
        config.summary = self.summary.get()
        if config.summary:
            if get_process_count('tensorboard.exe') == 0:
                subprocess.Popen(["tensorboard.exe", "--logdir",
                                  "d:\\myprogram\\ai\\zigoAc\\log"], shell=False)
            # C:\Program Files (x86)\Google\Chrome\Application\chrome.exe
            subprocess.Popen(["C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "http://localhost:6006"])
        else:
            if get_process_count('tensorboard.exe') > 0:
                subprocess.run(["taskkill", "/f", "/t", "/im", "tensorboard.exe"])
'''

    def stoptrain(self):
        config.running = False

    def test(self):
        selfplay.selftest(qp=self)
        # selfplay.elo()


class HelpWindow():
    def __init__(self, hpx):
        self.top = Toplevel()
        self.gui(hpx)
        self.top.mainloop()

    def gui(self, hpx):
        width = int(hpx * 0.8)
        height = int(hpx * 0.5)
        padx=int(hpx*0.005)
        pady=int(hpx*0.005)
        self.top.geometry('%dx%d+%d+%d' % (width, height, int(hpx* 0.3), int(hpx*0.2)))
        self.top.resizable(width=True, height=True)
        self.top.title("帮助")
        with open('help.txt', 'r') as f:
            helpstr = f.read()
        lbl1=ScrollText(None, self.top, font = ("宋体", 11))
        lbl1.grid(row = 0, column=0, stick=NSEW, padx=padx, pady=pady)
        lbl1.insert(END, helpstr)
        #self.top.columnconfigure(0, weight = 1)
        #self.top.rowconfigure(0, weight = 1)
