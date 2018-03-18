# -*- coding:utf-8 -*-
import os
import sys
import string
import re
from subprocess import Popen, PIPE
import threading
import time
import go

debug = 0 

def coords_to_sgf(size, board_coords):
    global debug
    
    board_coords = string.lower(board_coords)
    if board_coords == "pass":
        return ""
    if debug:
        print ("Coords: <" + board_coords + ">")
    letter = board_coords[0]
    digits = board_coords[1:]
    if letter > "i":
        sgffirst = chr(ord(letter) - 1)
    else:
        sgffirst = letter
    sgfsecond = chr(ord("a") + int(size) - int(digits))
    return sgffirst + sgfsecond



class GTP_connection(threading.Thread):

    def __init__(self, command):
        threading.Thread.__init__(self)
        self.command = command
        self.process = None
        self.running = False

    def run(self):
        try:
            self.process = Popen(self.command, stdin=PIPE, stdout=PIPE,
                stderr=PIPE, shell=True, universal_newlines=True)
        except:
            print ("excute {} failed".format(self.command))
            sys.exit(1)
        self.running = True
        print("程序已启动。")
        while not self.process.poll():
            #result = self.read()
            #if not result:
            #    break
            #self.onreceive(result)
            errline = self.process.stderr.readline()
            sys.stderr.write(errline)
            time.sleep(0.1)
        self.running = False
        print("程序已退出。")

    def read(self):
        result = ""
        line = self.process.stdout.readline()
        print(line)
        while line != "\n":
            result = result + line
            line = self.process.stdout.readline()
        if debug:
            sys.stderr.write("Reply: " + result + "\n")
        return result
        
    def send(self, cmd):
        if not self.running:
            return "程序还未运行。"
        global debug
        
        if debug:
            sys.stderr.write("GTP command: " + cmd + "\n")
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        result = self.read()
        # Remove trailing newline from the result
        if result[-1] == "\n":
            result = result[:-1]

        if len(result) == 0:
            return "ERROR: len = 0"
        if (result[0] == "?"):
            return "ERROR: GTP Command failed: " + result[2:]
        if (result[0] == "="):
            return result[2:]
        return result
        

class GTP_player:

    # Class members:
    #    connection     GTP_connection

    def __init__(self, command, name, color=-1):
        self.color = color
        self.player_type = 2
        self.passed = 0
        self.name = name
        self.connection = GTP_connection(command)
        self.connection.setDaemon(True)
        self.connection.start()
        while not self.connection.running:
            time.sleep(0.1)
        print(self.connection.send("name"))
        self.gtp_ready = True
        protocol_version = self.connection.send("protocol_version")
        if protocol_version[:5] != "ERROR":
            self.protocol_version = protocol_version
        else:
            self.protocol_version = "1"
        print ("gtp player ready! protocol version is " + self.protocol_version)

    def send(self, cmd):
        return self.connection.send(cmd)
        
    def is_known_command(self, command):
        return self.connection.send("known_command " + command) == "true"

    def genmove(self, color):
        if color[0] in ["b", "B"]:
            command = "black"
        elif color[0] in ["w", "W"]:
            command = "white"
        if self.protocol_version == "1":
            command = "genmove_" + command
        else:
            command = "genmove " + command

        return self.connection.send(command)

    def play(self, move, color):
        if color==go.BLACK:
            self.black(move)
        elif color==go.WHITE:
            self.white(move)

    def black(self, move):
        if self.protocol_version == "1":
            self.connection.send("black " + move)
        else:
            self.connection.send("play black " + move)

    def white(self, move):
        if self.protocol_version == "1":
            self.connection.send("white " + move)
        else:
            self.connection.send("play white " + move)

    def komi(self, komi):
        self.connection.send("komi {}".format(komi))

    def boardsize(self, size):
        self.connection.send("boardsize {}".format(size))
        if self.protocol_version != "1":
            self.connection.send("clear_board")

    def handicap(self, handicap, handicap_type="fixed"):
        if handicap_type == "fixed":
            result = self.connection.send("fixed_handicap {}".format(handicap))
        else:
            result = self.connection.send("place_free_handicap {}".format(handicap))

        return result.split(" ")

    def loadsgf(self, endgamefile, move_number):
        self.connection.send(string.join(["loadsgf", endgamefile, str(move_number)]))

    def list_stones(self, color):
        r = self.connection.send("list_stones " + color)
        return r.split(" ")

    def quit(self):
        r = self.connection.send("quit")
        print("执行退出{}命令，返回{}".format(self.name, r))
        return r
    
    def showboard(self):
        board = self.connection.send("showboard")
        if board and (board[0] == "\n"):
            board = board[1:]
        return board

    def get_random_seed(self):
        result = self.connection.send("get_random_seed")
        if result[:5] == "ERROR":
            return "unknown"
        return result

    def set_random_seed(self, seed):
        self.connection.send("set_random_seed " + seed)

    def get_program_name(self):
        return self.connection.send("name") + " " + \
               self.connection.send("version")

    def final_score(self):
        return self.connection.send("final_score")

    def score(self):
        return self.final_score()

    def cputime(self):
        if (self.is_known_command("cputime")):
            return self.connection.send("cputime")
        else:
            return "0"

    def analyse(self, sgffile=""):
        cmd = "analyse"
        if sgffile:
            cmd += " " + sgffile
        self.connection.send(cmd)
        #thsearch = threading.Thread(target=self.thread_analyse, args=(cmd,))
        #thsearch.setDaemon(True)
        #thsearch.start()

    def thread_analyse(self, cmd):
        self.connection.send(cmd)

    def stop_analyse(self):
        return self.connection.send("stop_analyse")
        
    
        
