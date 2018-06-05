#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

from tkinter import *
import configparser as cp
import go
import os
import logging
from logging.handlers import RotatingFileHandler

board_size = 19
running = True                  #控制循环用
background = 'pic/bj.png'
blackstone = 'pic/Black61.png'
whitestone = 'pic/White61.png'
blackbowls = 'pic/hqh.png'
whitebowls = 'pic/bqh.png'
lastmove = 'pic/last.png'
apps = {}

server = 'cgos.boardspace.net'
port9 = 6867
port13 = 6813
port19 = 6819
username = None
password = None
engine_name = 'ZiGo'
cgos_log = 'cgos.log'
log_level = 20
file_log_level = 10
sgf_dir = None
lz_dir = 'leelaz'

max_changes = 10
show_stats = True

# Log debug output to file
logger = None


def read_cfg(config_file='config'):
    global apps, board_size, sgf_dir, max_changes, log_level, file_log_level, show_stats
    global background,blackstone,whitestone,blackbowls,whitebowls,lastmove,lz_dir
    global server, port9, port13, port19, username, password, engine_name, cgos_log
    global logger

    cf = cp.ConfigParser()
    cf.read('config.ini', encoding="gb2312")
    background =  cf.get('THEME', 'background')
    blackstone = cf.get('THEME', 'blackstone')
    whitestone = cf.get('THEME', 'whitestone')
    blackbowls = cf.get('THEME', 'blackbowls')
    whitebowls = cf.get('THEME', 'whitebowls')
    lastmove = cf.get('THEME', 'lastmove')
    server = cf.get('CGOS', 'server')
    port9 = cf.getint('CGOS', 'port9')
    port13 = cf.getint('CGOS', 'port13')
    port19 = cf.getint('CGOS', 'port19')
    username = cf.get('CGOS', 'username')
    password = cf.get('CGOS', 'password')
    engine_name = cf.get('CGOS', 'engine_name')
    cgos_log = cf.get('CGOS', 'cgos_log')
    log_level = cf.getint('COMMON', 'log_level')
    file_log_level = cf.getint('COMMON', 'file_log_level')
    max_changes = cf.getint('COMMON', 'max_changes')
    sgf_dir = cf.get('COMMON', 'sgf_dir')
    lz_dir = cf.get('COMMON', 'lz_dir')
    board_size = cf.getint('COMMON', 'board_size')
    show_stats = cf.getboolean('COMMON', 'show_stats')

    apps={}
    lapps = cf["APPS"]
    for name, app in lapps.items():
        args = app.split(",")
        args[0] = os.path.realpath(args[0])
        app = ','.join(args)
        apps[name] = app

    logger = logging.getLogger("zigo")
    logger.setLevel(min(log_level, file_log_level))
    datefmt = '%m-%d %H:%M:%S'
    fmt = '%(asctime)s [%(levelname)s]: %(message)s'
    if file_log_level <= logging.DEBUG:
        fmt = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s'
    handler = logging.FileHandler("zigo.log", mode='w')   #Rotating, maxBytes=10*1024*1024,backupCount=3)
    handler.setLevel(file_log_level)
    formatter = logging.Formatter(fmt, datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Log info output to console
    #handler = logging.StreamHandler(sys.stdout)
    #handler.setLevel(config.log_level)
    #formatter = logging.Formatter("%(asctime)s: %(message)s")
    #handler.setFormatter(formatter)
    #config.logger.addHandler(handler)

def is_validate(content):        #如果你不加上==""的话，你就会发现删不完。总会剩下一个数字
    if content.isdigit() or (content==""):
        return True
    else:
        return False

def isfloat(s):
    m = re.match(r'^[0-9]*\.*[0-9]*$', s)
    if m or (s==""):
        return True
    return False

'''
class ConfigWindow():
    def __init__(self, wdpi, hdpi):
        self.top = Toplevel()
        self.gui(wdpi, hdpi)
        self.top.mainloop()

    def gui(self, wdpi, hdpi):
        global mtcs_width, mtcs_depth, mtcs_time, vresign
        width = int(wdpi * 7)
        height = int(hdpi * 5)
        padx=int(wdpi*0.05)
        pady=int(hdpi*0.05)
        self.top.geometry('%dx%d+%d+%d' % (width, height, wdpi* 3, hdpi*2))
        self.top.resizable(width=True, height=True)
        self.top.title("设置")
        self.vwidth=StringVar()
        self.vdepth=StringVar()
        self.vtime=StringVar()
        self.vvresign=StringVar()
        validate_fun=self.top.register(is_validate)#需要将函数包装一下，必要的
        lbl1=Label(self.top, text='蒙特卡洛搜索宽度：')
        self.txtwidth=Entry(self.top, textvariable = self.vwidth, validate='key', validatecommand=(validate_fun,'%P'))
        lbl2=Label(self.top, text='蒙特卡洛搜索深度：')
        self.txtdepth=Entry(self.top, textvariable = self.vdepth, validate='key', validatecommand=(validate_fun,'%P'))
        lbl3=Label(self.top, text='蒙特卡洛搜索时间：')
        self.txttime=Entry(self.top, textvariable = self.vtime, validate='key', validatecommand=(validate_fun,'%P'))
        lbl4=Label(self.top, text='投降的胜率阈值：')
        self.txtvresign=Entry(self.top, textvariable = self.vvresign, validate='key', validatecommand=(isfloat,'%P'))
        btnOk=Button(self.top, text='确定', command=self.save)
        btnCancel=Button(self.top, text='取消', command=self.cancel)
        lbl1.grid(row = 0, column=0, padx=padx, pady=pady)
        self.txtwidth.grid(row = 0, column=1, padx=padx, pady=pady)
        lbl2.grid(row = 1, column=0, padx=padx, pady=pady)
        self.txtdepth.grid(row = 1, column=1, padx=padx, pady=pady)
        lbl3.grid(row = 2, column=0, padx=padx, pady=pady)
        self.txttime.grid(row = 2, column=1, padx=padx, pady=pady)
        lbl4.grid(row = 3, column=0, padx=padx, pady=pady)
        self.txtvresign.grid(row = 3, column=1, padx=padx, pady=pady)
        btnOk.grid(row = 4, column = 0, padx=padx*4, pady=pady)
        btnCancel.grid(row = 4, column = 1, padx=padx*4, pady=pady)
        self.top.columnconfigure(1, weight = 1)
        self.top.rowconfigure(3, weight = 1)
        self.vwidth.set(str(mtcs_width))
        self.vdepth.set(str(mtcs_depth))
        self.vtime.set(str(mtcs_time))
        self.vvresign.set(str(vresign))

    def save(self, config_file=None):
        global mtcs_width, mtcs_depth, mtcs_time, vresign, last_config
        if config_file:
            last_config = config_file
        else:
            config_file = last_config
        config_file = 'config/%s.ini' % (config_file)
        mtcs_width = int(self.vwidth.get())
        mtcs_depth = int(self.vdepth.get())
        mtcs_time = int(self.vtime.get())
        vresign = float(self.vvresign.get())
        cf = cp.ConfigParser()
        cf.read(config_file)
        cf.set('MTCS', 'width', str(mtcs_width))
        cf.set('MTCS', 'depth', str(mtcs_depth))
        cf.set('MTCS', 'time', str(mtcs_time))
        cf.set('PLAY', 'vresign', str(vresign))
        with open(config_file,"w") as f:
            cf.write(f)
        shutile.copyfile(config_file, 'config/config.ini')
        cf0=cp.ConfigParser()
        cf0.read('config/config.ini')
        cf0.set('PLAY', 'last_config', last_config)
        with open('config/config.ini',"w") as f:
            cf0.write(f)
        self.top.destroy()

    def cancel(self):
        self.top.destroy()
'''
