#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

import sys
#import sqlite3
import go
import board
#import dataset
import numpy as np
import time

ZOBRIST_EMPTY = 0x0
ZOBRIST_BLACKTOMOVE = 0x2BCDABCDABCDABCD

ZOBRIST = []

def init_zobrist():
    global ZOBRIST
    maxsq = go.MAX_BOARD*go.MAX_BOARD
    for i in range(1,6):
        zobrist = []
        for j in range(1, maxsq+1):
            zobrist.append(((i*(maxsq+21)+j) << 44) + (((i+11)*maxsq+j) << 20) + i * maxsq + j);
        ZOBRIST.append(zobrist)

def get_hash(board):
    return calc_hash(board.stones, board.to_move, board.ko)

def get_ko_hash(board):
    return calc_hash(board.stones, None, None)

def calc_hash(stones, to_move, ko):
    res = ZOBRIST_EMPTY
    pad = int((go.MAX_BOARD-go.N)/2)
    for i in range(go.MAX_BOARD):
        for j in range(go.MAX_BOARD):
            index = i*go.MAX_BOARD+j
            if i>=pad and i<go.N+pad and j>=pad and j<go.N+pad:
                c = stones[(i-pad)*go.N+(j-pad)]
                res = res ^ (ZOBRIST[c][index])
            else:
                res = res ^ (ZOBRIST[go.BORDER][index])
    if to_move is None:
        return res
    if to_move == go.BLACK:
        res = res ^ ZOBRIST_BLACKTOMOVE
    if ko is not None and ko>0 and ko<go.N2:
        res = res ^ (ZOBRIST[go.KO][get_index(ko)])
    return res


def get_index(move):
    y,x=divmod(move, go.N)
    pad = int((go.MAX_BOARD-go.N)/2)
    return (y+pad)*(go.MAX_BOARD)+x+pad

init_zobrist()
"""
class SqliteDb:
    def __init__(self, file="zigo"):
        file = file+"-"+str(go.N)+".db"
        self.conn = sqlite3.connect(file, isolation_level=None)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS train(
                            points REAL NOT NULL,tomove INT,ko INT DEFAULT(-1),
                            hash INTEGER UNIQUE ON CONFLICT IGNORE NOT NULL,
                            stones0 TEXT,stones1 TEXT,stones2 TEXT,stones3 TEXT,
                            stones4 TEXT,stones5 TEXT,stones6 TEXT,stones7 TEXT)''')
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS index_hash ON train(hash ASC);")
        self.cursor = self.conn.cursor()

    def close(self):
        self.cursor.close()
        self.conn.close()

    def get_count(self):
        self.cursor.execute("select count(*) from train")
        row = self.cursor.fetchone()
        if row:
            return row[0]
        return 0

    def save(self, data):
        try:
            self.conn.execute("BEGIN TRANSACTION")
            binds = [d.points,d.board.to_move,d.board.ko,d.board.hash]
            for d in data:
                for i in range(8):
                    stones = d.board.get_past_stones(i)
                    binds.append(adapter_list(stones))
                self.conn.execute('''INSERT OR IGNORE INTO train values
                        (?,?,?,?,?,?,?,?,?,?,?,?)''',binds)

        except sqlite3.Error as e:
            self.conn.rollback()
            print("存数据库出错！{}".format(e))
        except Exception as e:
            self.conn.rollback()
            print("存数据库出错：{}".format(e))
        else:
            self.conn.execute("COMMIT")

    def search(self, board):
        hashs=[]
        for i in range(8):
            ss = rotate_board(board.stones, i)
            ko = rotate_move(board.ko, i)
            hash = calc_hash(ss, board.to_move, ko)
            hashs.append(hash)
        sqlstr = '''SELECT points,stones0,hash,rowid FROM train WHERE hash IN(?,?,?,?,?,?,?,?);'''
        try:
            self.cursor.execute(sqlstr, hashs)
        except sqlite3.Error as e:
            print("读数据库出错！{}".format(e), file=sys.stderr)
            #self.close()
            return -99999.0
        except Exception as e:
            print("读数据库出错：{}".format(e), file=sys.stderr)
            return -99999.0
        row = self.cursor.fetchone()
        if row:
            return row[0]
        return -99999.0

    def load_data(self, num=10000):
        cnt = self.get_count()
        start_time = time.time()
        sqlstr = '''SELECT points,tomove,ko,stones0,stones1,stones2,stones3,
            stones4,stones5,stones6,stones7 FROM train WHERE rowid IN 
            (SELECT rowid FROM train ORDER BY RANDOM() LIMIT ?)'''
        #startid = max(0, cnt-num)
        try:
            self.cursor.execute(sqlstr, [num])
        except sqlite3.Error as e:
            print("读数据库出错！{}".format(e), file=sys.stderr)
            return [None]
        except Exception as e:
            print("读数据库出错：{}".format(e), file=sys.stderr)
            return [None]
        rows = self.cursor.fetchall()
        print("读数据库{}条记录耗时：{}。".format(len(rows), time.time()-start_time))
        for row in rows:
            points = row[0]
            tomove = row[1]
            ko=go.unflatten_coords(row[2])
            stoneses = [convert_list(b) for b in row[3:]]
            states = dataset.get_feature_from_stones(stoneses, tomove)
            pos=board.Board(stoneses[0], ko=ko, to_move=tomove)
            moves=pos.get_moves()
            move_values = np.zeros([go.N * go.N+2], dtype=np.float32)
            if not moves:
                moves.append(go.PASS)
                moves[go.N*go.N] = 1.0
            else:
                for move in moves:
                    bd = pos.try_move(move)
                    nextpoints = self.search(bd)
                    if nextpoints<-9999:
                        continue
                    ind = go.flatten_coords(move)
                    move_values[ind] = nextpoints-points
            move_values = dataset.softmax(move_values)
            points = points/(go.N*go.N)
            yield ([states.tolist()], [move_values.tolist()], [points])
        
def adapter_list(list):
    l = (list+1).flatten().tolist()
    b = bytes(l)
    s = bytes.decode(b)
    return s

def convert_list(s):
    b = str.encode(s)
    l = np.array(list(b))-1
    return l.reshape(go.N,go.N)

def rotate_board(stones, ind):
    if ind==0:
        return stones
    if ind<4:
        return np.rot90(stones, k=ind)
    elif ind>=4:
        flip = np.fliplr(stones)
        return np.rot90(flip, k=ind-4)

def rotate_move(move,ind):
    if not move:
        return move
    y=move[0]
    x=move[1]
    if ind < 4:
        newx = x
        newy = y
        for i in range(ind):
            z=newx
            x=newy
            y=z
            newx = x
            newy = go.N - y - 1
    else:
        newx = go.N - x - 1;
        newy = go.N - y - 1;
        for i in range(ind-4):
            z=newx
            x=newy
            y=z
            newx = x
            newy = go.N - y - 1
    return newy,newx
"""