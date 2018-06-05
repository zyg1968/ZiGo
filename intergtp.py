#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

import os, sys, string, re, threading, time
from subprocess import Popen, PIPE, STDOUT
import config
import go
import player
import logging
from logging.handlers import RotatingFileHandler

ANALYSER, PLAYER, SELF_PLAY = 0, 1, 2

def coords_to_sgf(size, board_coords):
    board_coords = string.lower(board_coords)
    if board_coords == "pass":
        return ""
    self.logger.debug("Coords: <" + board_coords + ">")
    letter = board_coords[0]
    digits = board_coords[1:]
    if letter > "i":
        sgffirst = chr(ord(letter) - 1)
    else:
        sgffirst = letter
    sgfsecond = chr(ord("a") + int(size) - int(digits))
    return sgffirst + sgfsecond

class MyIO:
    def __init__(self):
        self.lines = []
        self.line = ""
        self.moves = []

    def write(self, s):
        self.line += s
        if '\n' in s:
            self.lines.append(self.line)
            self.line = ''
        filtstr= r"\d+ \([BW] ([A-Z]\d+)\)[\s\n]"
        ms = re.findall(filtstr, self.line)
        if ms:
            if len(self.lines)>0 and '\n' in self.lines[-1]:
                self.lines.append(self.line)
            else:
                self.lines[-1] += self.line
            self.line = ''
            self.moves = []
            for m in ms:
                move = go.get_coor_from_gtp(m)
                self.moves.append(move)
                sys.stderr.write("New move\n")

class GTP_connection(threading.Thread):

    def __init__(self, command, connect_type=PLAYER, log=None):
        threading.Thread.__init__(self)
        self.command = command
        self.process = None
        self.running = False
        self.values = None
        self.ponder = False
        self.is_dumping = False
        self.connect_type = connect_type
        self.visits = 0
        self.lines = []
        self.line = ""
        self.moves = []
        self.log_init(log)

    def log_init(self, log):
        if log is None:
            log = "gtp"
        self.logger = logging.getLogger(log)
        self.logger.setLevel(min(config.file_log_level, config.log_level))
        datefmt = '%m-%d %H:%M:%S'
        fmt = '%(asctime)s [%(levelname)s]: %(message)s'
        if config.file_log_level <= logging.DEBUG:
            fmt = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s'
        handler = logging.FileHandler(log + ".log", mode='w') #Rotating, maxBytes=10*1024*1024,backupCount=3)
        handler.setLevel(config.file_log_level)
        formatter = logging.Formatter(fmt, datefmt)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


    def run(self):
        path = os.path.realpath(self.command[0])
        self.command[0] = path
        dir,fn = os.path.split(path)
        try:
            self.process = Popen(self.command, stdin=PIPE, stdout=PIPE,
                stderr=PIPE, cwd=dir, shell=True, universal_newlines=True)
        except Exception as e:
            self.logger.exception('popen')
            exit(1)
        self.running = True
        self.logger.info("程序{}已启动。".format(fn))
        while not self.process.poll():
            if self.connect_type == SELF_PLAY:
                c = self.process.stdout.read(1)
                self.self_play(c)
            else:
                errline = self.process.stderr.readline()
                if errline != "" and errline != "\n":
                    self.logger.debug(errline.replace('\n', '', 1))
                self.analyse_values(errline)
                #if self.connect_type == PLAYER and config.show_stats:
                #    sys.stderr.write(errline)
            time.sleep(0.1)
        self.running = False
        print("程序{}已退出。".format(fn))

    def read(self):
        result = ""
        line = self.process.stdout.readline()
        while line != "\n":
            result = result + line
            line = self.process.stdout.readline()
        return result
        
    def send(self, cmd):
        if not self.running:
            return "程序还未运行。"
        self.logger.debug('Gtp send: ' + cmd)
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        result = self.read()
        self.logger.debug('Gtp reply: ' + result)
        # Remove trailing newline from the result
        if result[-1] == "\n":
            result = result[:-1]
        if len(result) == 0:
            result = "ERROR: len = 0"
        elif (result[0] == "?"):
            result = "ERROR: GTP Command failed: " + result[2:]
        elif (result[0] == "="):
            result = result[2:]
        return result

    def write(self, cmd):
        if not self.running:
            return "程序还未运行。"
        self.logger.debug('Gtp write: ' + cmd)
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()


    def self_play(self, s):
        #1 (B D17) 2(W Q16)
        self.line += s
        if '\n' in s:
            self.lines.append(self.line)
            self.logger.debug(self.line.replace('\n', '', 1))
            self.line = ''
        filtstr= r"\d+ \([BW] ([A-Z]\d+)\)[\s\n]"
        ms = re.findall(filtstr, self.line)
        if ms:
            if len(self.lines)>0 and '\n' in self.lines[-1]:
                self.lines.append(self.line)
            else:
                self.lines[-1] += self.line
            self.line = ''
            self.moves = []
            for m in ms:
                move = go.get_coor_from_gtp(m)
                self.moves.append(move)
                sys.stderr.write("New move\n")

    def analyse_values(self, s):
        if 'ponder stats:' in s:
            self.ponder = True
            return
        if not self.is_dumping:
            fs1=r"start dump stats, all visits count: (\d+)\n"
            m1 = re.match(fs1, s)
            if m1:
                self.values = {}
                self.visits = int(m1.group(1))
                self.is_dumping = True
            return
        filtstr= r"\s*([A-Z]\d{1,2})\s\->\s+(\d+)\s*\(V:\s*([\d\.]+)\%\)\s\(N:\s*[\d\.]+\%\)\sPV:\s(.+)\n"   #"(Playouts: \d+, Win: ([\d\.]+)\%, PV: ([A-Z]\d{1,2})[\s\n])"|
        m = re.match(filtstr, s)
        if m:
            move = go.get_coor_from_gtp(m.group(1))
            wr = float(m.group(3))
            if self.ponder:
                wr = 100.0 - wr
            data = go.AnalyseData(move, self.visits, int(m.group(2)), wr, m.group(4))
            self.values[move] = data
        elif "dump state end: " in s:
            self.ponder = False
            self.is_dumping = False
            if self.values:
                if self.connect_type == ANALYSER:
                    sys.stderr.write("Analyse_Changed\n")
                else:
                    sys.stderr.write("Values_Changed\n")
            self.visits = 0

    def send_signal(self, signal):
        self.process.send_signal(signal)

class GTPPlayer(player.Player):
    # Class members:
    #    connection     GTP_connection

    def __init__(self, command, name, color=go.BLACK, is_analyser=False, log=None):
        super().__init__(name, player_type = player.GTP, color=color)
        self.connection = GTP_connection(command, ANALYSER if is_analyser else PLAYER, log)
        self.connection.setDaemon(True)
        self.connection.start()
        while not self.connection.running:
            time.sleep(0.1)
        n = self.connection.send("name")
        self.gtp_ready = True
        protocol_version = self.connection.send("protocol_version")
        if protocol_version[:5] != "ERROR":
            self.protocol_version = protocol_version
        else:
            self.protocol_version = "1"
        config.logger.info("{}方 {} gtp ready! protocol version is {}".format(
            go.get_color_str(color), n, self.protocol_version))

    def send(self, cmd):
        if self.connection.running:
            #config.logger.debug('Send: {}'.format(cmd))
            r = self.connection.send(cmd)
            #config.logger.debug('Reply: {}'.format(r))
            return r
        config.logger.error('connection is breaked!')
        return "error: connection is breaked!"

    def write(self, cmd):
        if self.connection.running:
            #config.logger.debug('Write: {}'.format(cmd))
            self.connection.write(cmd)
        
    def is_known_command(self, command):
        return self.send("known_command " + command) == "true"

    def genmove(self, color):
        if color==go.BLACK:
            command = "black"
        else:
            command = "white"
        if self.protocol_version == "1":
            command = "genmove_" + command
        else:
            command = "genmove " + command

        return self.send(command)

    @property
    def values(self):
        return self.connection.values

    def play(self, move, color):
        if color==go.BLACK:
            self.black(move)
        elif color==go.WHITE:
            self.white(move)

    def black(self, move):
        if self.protocol_version == "1":
            self.send("black " + move)
        else:
            self.send("play black " + move)

    def white(self, move):
        if self.protocol_version == "1":
            self.send("white " + move)
        else:
            self.send("play white " + move)

    def komi(self, komi):
        return self.send("komi {}".format(komi))

    def boardsize(self, size):
        self.send("boardsize {}".format(size))
        if self.protocol_version != "1":
            self.clear_board()

    def clear_board(self):
        return self.send("clear_board")

    def set_time(self, time_settings):
        if not time_settings:
            return
        player.Player.set_time(self, time_settings)
        return self.send("time_settings {}".format(time_settings))

    def time_left(self, color, msec, stones=0):
        return self.send("time_left {} {} {}".format(color, msec, stones))

    def handicap(self, handicap, handicap_type="fixed"):
        if handicap_type == "fixed":
            result = self.send("fixed_handicap {}".format(handicap))
        else:
            result = self.send("place_free_handicap {}".format(handicap))

        return result.split(" ")

    def loadsgf(self, endgamefile, move_number):
        return self.send(" ".join(["loadsgf", endgamefile, str(move_number)]))

    def list_stones(self, color):
        r = self.send("list_stones " + color)
        return r.split(" ")

    def undo(self):
        return self.send("undo")

    def quit(self):
        self.is_quit = True
        return self.send("quit")
    
    def showboard(self):
        board = self.send("showboard")
        if board and (board[0] == "\n"):
            board = board[1:]
        return board

    def get_random_seed(self):
        result = self.send("get_random_seed")
        if result[:5] == "ERROR":
            return "unknown"
        return result

    def set_random_seed(self, seed):
        self.send("set_random_seed " + seed)

    def get_program_name(self):
        return self.send("name") + " " + \
               self.send("version")

    def final_score(self):
        return self.send("final_score")

    def score(self):
        return self.final_score()

    def clear_dead(self):
        stones_text = self.send("clear_dead")
        stones = []
        for c in stones_text:
            ci = int(c, 16)
            ci0 = ci>>2
            if len(stones)<go.N2:
                stones.append(ci0 & 0x03)
            if len(stones)<go.N2:
                stones.append(ci & 0x03)
        return stones

    def cputime(self):
        if (self.is_known_command("cputime")):
            return self.send("cputime")
        else:
            return "0"

    def analyse(self, sgffile=""):
        cmd = "analyse"
        if sgffile:
            cmd += " " + sgffile
        return self.send(cmd)
        #thsearch = threading.Thread(target=self.thread_analyse, args=(cmd,))
        #thsearch.setDaemon(True)
        #thsearch.start()

    def thread_analyse(self, cmd):
        return self.send(cmd)

    def stop_analyse(self):
        return self.send("stop_analyse")
        
    
        
