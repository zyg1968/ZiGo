#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

import threading, time, sys
import go
import config

policy_names=['policy', 'ramdom', 'mtcs']
PLAYER, POLICY, GTP, CGOS = 0, 1, 2, 3

class Player():
    def __init__(self, name, player_type=PLAYER, time=20, color=1):
        self.name=name
        self.player_type = player_type
        self.time=time
        self.countdown = 0
        self.counttimes = 0
        self.color = color
        self.is_quit = False

    def genmove(self, color = None):
        pass
    def play(self, coord, color = None):
        pass
    def undo(self):
        pass
    def komi(self, komi):
        pass
    def boardsize(self, size):
        pass
    def clear_board(self):
        pass
    def quit(self):
        pass
    def score(self):
        pass
    def handicap(self, handicap, handicap_type="fixed"):
        pass
    def set_time(self, time_settings):
        ts = time_settings.split(' ')
        if len(ts)>0:
            self.time = int(ts[0])
        if len(ts)>1:
            self.countdown = int(ts[1])
        if len(ts)>2:
            self.counttimes = int(ts[2])

    def time_left(self, color = None, msec = 0):
        self.time = int(msec/1000)

def play(player1, player2, qp=None, start_color=go.BLACK):
    thplay = PlayThread('下棋', qp, player1, player2, start_color=start_color)
    thplay.setDaemon(True)
    thplay.start()

class PlayThread (threading.Thread):
    def __init__(self, name, qp, player1, player2, start_color=go.BLACK):
        threading.Thread.__init__(self)
        self.name = name
        self.qp=qp
        self.running=False
        self.start_color = start_color
        self.player1 = player1
        self.player2 = player2
        self.qp.start(self.qp.board)

    def run(self):
        config.logger.info("开启线程： " + self.name)
        self.running=True
        self.playloop()
        self.running=False
        if self.player1.player_type == GTP:
            self.player1.quit()
        if self.player2.player_type == GTP:
            self.player2.quit()
        config.logger.info("退出线程： " + self.name)

    def playloop(self):
        c = go.oppo_color(self.start_color)
        while(self.running and not self.qp.board.is_gameover):
            #c = self.qp.board.to_move
            c = go.oppo_color(c)
            player = self.player1 if c==self.player1.color else self.player2
            opponent = self.player2 if c==self.player1.color else self.player1
            gtpmove = ""
            if player.player_type == POLICY:
                coor, win_rate = player.get_move(self.qp.board, caps)
                self.qp.board.win_rate = win_rate
                gtpmove = go.get_cmd_from_coor(coor)
                #coor=self.gomanager.get_coor_from_vertex(vertex)  #从上倒下，左到右，0开始
            elif player.player_type == GTP:
                gtpmove = player.genmove(c)
                #coor= go.get_coor_from_gtp(coorstr)
            elif player.player_type == PLAYER:
                self.qp.clicked=None
                while(self.qp.clicked is None):
                    time.sleep(0.1)
                coor=self.qp.clicked
                self.qp.clicked = None
                gtpmove = go.get_cmd_from_coor(coor)
            else:
                config.logger.error("Not know player.")
            gtpmove = gtpmove.upper()
            if gtpmove == "RESIGN":
                self.qp.board.points = go.N*go.N+1 if c==go.WHITE else -(go.N*go.N+1)
                self.qp.board.result = self.qp.board.get_result() 
                break
            elif gtpmove == "UNDO":
                self.qp.board.undo()
                opponent.undo()
                self.qp.board.undo() 
                opponent.undo()
                self.qp.update()
                c = go.oppo_color(c)
            else:
                if opponent.player_type == GTP:
                    opponent.play(gtpmove, c)
                if player.player_type != PLAYER:
                    self.qp.play(gtpmove, c)
            self.running=config.running

        result=""
        if self.qp.board.result is None:
            self.qp.board.points=self.qp.board.final_score()
            self.qp.board.result = self.qp.board.get_result()
        result = go.result_str(self.qp.board.result)
        msg = "对局结束。%s" % (result)
        config.logger.info(msg)
        #self.qp.show_message(msg=msg, status=msg)

