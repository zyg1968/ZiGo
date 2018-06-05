#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

import socket
import sys, traceback, time
import os.path
import string
import random

import go
import board
import intergtp
import sgfparser
import config
import player

class CGOSClientError(Exception):
    def _init__(self, msg):
        self.msg = msg
    def _str__(self):
        return repr(self.msg)

class CGOSConnector():
    CLIENT_ID = "ZiGo" #cgosGtp 0.98 alpha - engine client for CGOS [Windows-x86] by Don Dailey
    
    # How often to output stats, etc., in seconds 
    TIME_CHECKPOINT_FREQUENCY = 60 * 30
    
    def __init__(self, qp=None, player=None, board_size=19):
        '''
        Initialise the client, without connecting anything yet
        '''
        self.player = player
        self.qp = qp
        self.finished = False              # Should the main loop quit
        self.socketfile = None
        self.port = config.port9 if board_size==9 else  \
            (config.port13 if board_size==13 else config.port19)
        
        self.gameInProgress = False        # Currently between setup and gameover?
        self.wins = 0                  # Stats about how many games were won/lost during this session
        self.losts = 0
        self.color = go.BLACK
        self.id = None
        
        self.timeStarted = time.localtime()    # Will not change
        self.timeCheckPoint = time.localtime() # Last time checkpoint for outputting stats, mail, etc.
            
    @property
    def is_play(self):
        return True if self.player else False

    def connect(self):
        config.logger.info("Attempting to connect to server '" + config.server + "', port " + str(self.port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            self.socket.connect((config.server, self.port))
        except Exception as e:
            raise CGOSClientError("Connection failed: " + str(e))
            
        self.socketfile = self.socket.makefile(mode='rw')
        config.running = True
        self.finished = False
        config.logger.info("Connected")
       
    def disconnect(self):
        config.logger.info("Disconnecting")
        if self.socketfile is not None:
            self.socketfile.close()
            self.socket.close()
            self.socketfile = None

    def respond(self, message):        
        if self.socket is not None:
            config.logger.debug("Responding: " + (message if message != config.password else '******')) 
            self.socketfile.write(message)
            self.socketfile.flush()

    def show_match(self, id):
        self.respond("observe "+id)

    def handle_info(self, parameters):
        ''' Event handler: "info". Ignored. '''
        config.logger.info("Server info: " + (" ".join(parameters)))
        self.checkTimeCheckpoint()
    
    def handle_protocol(self, parameters):
        ''' Event handler: "protocol" command. No parameters. '''
        info = "v1 "
        if self.is_play:
            info = "e1 "
        info += config.engine_name # CGOSConnector.CLIENT_ID
        self.respond(info)
                      
    def handle_username(self, parameters):
        ''' Event handler: "username" command. No parameters. '''
        self.respond(config.username)

    def handle_password(self, parameters):
        ''' Event handler: "password" command. No parameters. '''
        self.respond(config.password)

    def handle_match(self, parameters):
        #[id,date,time,board_size,komi,white,black,result]
        datas = []
        if len(parameters)>7:
            datas.append(parameters[0])
            datas.append(parameters[1])
            datas.append(parameters[2])
            datas.append(parameters[5])
            datas.append(parameters[6])
            datas.append(parameters[7])
        else:
            datas = parameters
            for i in range(7-len(datas)):
                datas.append("")
        if self.qp.cgoswin and self.qp.cgoswin.closed:
            self.qp.cgoswin.show()
        self.qp.cgoswin.add(datas, getattr(self, "show_match"))

    def handle_update(self, parameters):
        if parameters is None:
            config.logger.error("handle update parameters is none.")
            return
        id = int(parameters[0])
        if id!=self.id:
            return
        parlen = len(parameters)
        if '+' in parameters[1] and parlen<3 and self.qp:
            self.qp.board.result = parameters[1]
            return
        for i in range(1, parlen, 2):
            coord = parameters[i]
            self.color = go.oppo_color(self.color)
            t = '0'
            if i+1<parlen:
                t = parameters[i+1]
            self.handle_play(['b' if self.color==go.BLACK else 'w', coord, t])
        
    def handle_setup(self, parameters):
        ''' 
        Event handler: "setup" command to prepare for game.
        Expects the following parameters:        
          1- Game id
          2-
          3-
          4- Board size
          5- Komi
          6- Program A name, with optional rating, e.g. "program(1800)"
          7- Program B name, with optional rating
          8- Game time per player in msec
        
        The parameters may be followed by an alternating list of moves/time pairs, 
        starting with black, to place on the board.
            e1 setup 406484 19 7.5 900000 fuego1604_10k(2203) ZiGo(2962?)  
        Example: v1 setup 405053 - - 19 7.5 Perseus-8(3792) LZ-F192(3573?) 900000 C4 887634 q4 898779   
        '''
        if parameters is None:
            config.logger.error("handle setup parameters is none.")
            return
        paralen = len(parameters)
        setuplen = 6 if self.is_play else 8
        indexs = [0,1,2,3,4,5] if self.is_play else [0,3,4,7,5,6]
        if (paralen < setuplen): raise CGOSClientError("'setup' command requires at least 8 parameters")
        
        self.gameInProgress = True
        
        # Parse the parameters, and cut apart the rank and player names
        self.id = int(parameters[0])
        boardsize = int(parameters[indexs[1]])
        komi = float(parameters[indexs[2]])
        gameTimeMSec = int(parameters[indexs[3]])//1000
        programA = parameters[indexs[4]]
        programB = parameters[indexs[5]]
        
        programARank = ""
        programBRank = ""
        if "(" in programA:
            programARank = programA[programA.find("(") : programA.rfind(")")].strip("()")
            programA = programA[:programA.find("(")]
        if "(" in programB:
            programBRank = programB[programB.find("(") : programB.rfind(")")].strip("()")
            programB = programB[:programB.find("(")]        
            
        if self.player:
            self.player.color = go.BLACK
            if config.username in programA:
                self.player.color = go.WHITE
        # Log some information
        if paralen > setuplen:
            config.logger.info("This is a restart. Catching up " + str((len(parameters)-setuplen) // 2) + " moves")
        
        # Set up the engine through GTP. Also observer, if registered  
        if self.is_play:
            self.player.boardsize(boardsize)
            self.player.komi(komi)
            self.player.clear_board()
            #self.player.set_time("{} {} {}".format(gameTimeMSec,0,25))
        
        if self.qp is not None:
            if boardsize!=go.N:
                go.set_board_size(boardsize)
            self.qp.player1_name['text'] = "{}\n({})".format(programB, programBRank)
            self.qp.player2_name['text'] = "{}\n({})".format(programA, programARank)
            self.qp.clear()
            self.qp.board = board.Board()
            self.qp.board.komi=komi
            self.qp.set_time("{} {} {}".format(gameTimeMSec,0,0))
        
        # If there are more than 8 parameters, we need to notify the engine of moves to catch up on
        if paralen > setuplen:
            self.color = go.BLACK            
            for i in range(setuplen,paralen,2):
                gtpmove = parameters[i].upper()
                t=0
                if i+1<paralen:
                    t = parameters[i+1]
                if gtpmove[1]=='+' and self.qp:
                    self.qp.board.result = gtpmove
                    if not self.is_play:
                        self.quit()
                    if gtpmove[2:] != "RESIGN":
                        continue
                    else:
                        gtpmove = gtpmove[2:]
                self.handle_play(['b' if self.color == go.BLACK else 'w', gtpmove, t])
                self.color = go.oppo_color(self.color)
            
    def handle_play(self, parameters):
        '''
        Event handler: "play" command. Expects:
          - GTP colour
          - GTP coordinate
          - Time left in msec
        '''
        if (len(parameters) != 3): raise CGOSClientError("'play' command requires 3 parameters")
        
        colour = parameters[0].lower()
        color = go.BLACK if colour=='b' else go.WHITE
        coord = parameters[1].upper()
        msec = int(parameters[2])//1000

        if self.is_play:
            self.player.play(coord, color)
            self.player.time_left(colour, msec)
        
        if self.qp is not None:
            self.qp.play(coord, color)
            self.qp.time_left(color, msec)
                    
    def handle_genmove(self, parameters):
        '''
        Event handler: "genmove". Expects:
          - GTP colour
          - Time left in msec
        '''
        if (len(parameters) != 2): raise CGOSClientError("'play' command requires 2 parameters")
        
        colour = parameters[0].lower()
        color = go.BLACK if colour[0]=="b" else go.WHITE
        msec = int(parameters[1])//1000
        if self.is_play:
            gtpmove = self.player.genmove(color)
            self.respond(gtpmove)
            self.player.time_left(colour, msec)
        self.qp.play(gtpmove, color)
        self.qp.time_left(color, msec)
                
    def handle_gameover(self, parameters):
        '''
        Event handler: "gameover". Expects:
          - A date
          - The result (unparsed) e.g. "B+Resign" gameover 2018-04-03 W+Time
        '''
        id = 0
        if self.is_play:
            overtime = parameters[0]
        else:
            id = int(parameters[0])

        if not self.is_play and id>0 and id!=self.id:
            return
        result = parameters[1]        
        config.logger.info("Game over. Result: " + result)

        winer = (go.BLACK if result.upper()[0]=='B' else go.WHITE)
        if self.player:
            res = result.lower()[2:]
            if res == 'time' or res == 'illegal':
                res = 'resign'
            if res == 'resign':
                self.player.play(res, go.oppo_color(winer))
            #self.handle_play([result.lower()[0], result.lower()[2:], '0']);
            if self.player.color == winer:
                config.logger.info("Local engine won :-)")
                self.wins += 1
            else:
                config.logger.info("Local engine lost :'(")
                self.losts += 1
        
        self.gameInProgress = False
        if self.qp.board:
            self.qp.board.result = result
        if self.is_play:
            self.check_resume()
            if not(self.finished): self.respond("ready")
            self.checkTimeCheckpoint()
        else:
            self.finished=True
        
    def handlerloop(self):
        while not(self.finished) and config.running and self.socketfile:            
            line = self.socketfile.readline()
            line = line.strip()
            config.logger.debug("Server sent: " + line)
            
            if len(line) == 0:
                config.logger.debug("Empty line received from CGOS server")
                continue
            
            if line.startswith("Error:"):
                config.logger.error("CGOS Error: " + line[6:])
                #if "You are already logged on" in line:
                #    self.disconnect()
                self.finished = True
                return
                
            splitline = line.split(None, 1)
            
            commandHandler = "handle_" + splitline[0]            
            try:
                handler = getattr(self, commandHandler)
            except AttributeError:
                #config.logger.exception('AttributeError')
                config.logger.error("Unsupported CGOS command, '"+splitline[0]+"'")
            #    raise CGOSClientError("Unsupported command: " + splitline[0])
            else:
                parameters = []
                if (len(splitline) > 1): parameters = splitline[1].split()
                try:
                    result = handler(parameters)
                except CGOSClientError as e:
                    raise CGOSClientError(str(e))
            
            if not(self.gameInProgress): self.check_resume()
                
    def check_resume(self):
        '''
        Check if the kill file exists and set _finished to true if yes.
        '''
        self.finished = not config.running
        if self.finished:
            config.logger.info("Shutting down connection and engines.")
    
    def checkTimeCheckpoint(self):
        '''
        Check if the last time checkpoint was more than half an hour away, and
        perform maintenance tasks if necessary (information output, etc).
        
        This should not be called from time-sensitive parts like genmove, but from
        info messages, gameover, etc.
        '''
        currentTime = time.mktime(time.localtime())
        duration = currentTime - time.mktime(self.timeCheckPoint)
 
        if duration > CGOSConnector.TIME_CHECKPOINT_FREQUENCY:            
            self.timeCheckPoint = time.localtime()
            
            duration = currentTime - time.mktime(self.timeStarted)
            config.logger.info("Client up for " + str(int(duration)//3600) + " hours, " +
                             str((int(duration)//60)%60) + " mins, " + str(int(duration)%60) + " seconds. " + 
                             "Local engines won " + str(self.wins) + " games, lost " + 
                             str(self.losts) + ".")
        
    def isConnected(self):
        return self.socketfile is not None
    
    def mainloop(self):
        self.finished = False
        while not(self.finished):
            connected = False
            retries = 1 
            while not(connected):
                if self.finished:
                    break
                try:
                    self.connect()
                    connected = True
                except:
                    config.logger.error("Could not connect to " + config.server +". Will try again.")
                    time.sleep(30 + int(random.random()*5))
                    retries += 1
                        
            try:
                self.handlerloop()
            except socket.error:
                config.logger.error("Socket error. CGOS connection lost.")
                self.disconnect()
            except CGOSClientError as e:
                config.logger.exception('CGOSClientError')
                return
            #except Exception as e:
            #    config.logger.error("error: " + str(e))
            #    return
        self.respond("quit")

    def quit(self):
        config.logger.info("Shutting down CGOS connection")
        self.finished = True
        if self.socketfile is not None:
            self.socketfile.close()
            self.socket.close()
            self.socketfile = None
            self.socket = None
        
def start_cgos(client):
    try:
        client.mainloop()
    finally:
        if client.socketfile:
            client.quit()
    client = None
    config.logger.info("CGOS线程关闭。")



