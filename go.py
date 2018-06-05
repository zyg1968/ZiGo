#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

from collections import namedtuple
import copy
import itertools
import gtp
import numpy as np
import random

MAX_BOARD = 19

EMPTY, BLACK, WHITE, BORDER, KO, FILL, UNKNOWN = range(0, 7)

UNDO = -3

class IllegalMove(Exception): pass

# these are initialized by set_board_size
N = None
N2 = None
ALL_COORDS = []
EMPTY_BOARD = None
NEIGHBORS = {}
DIAGONALS = {}
COLLUM_STR = ''
PASS = None
RESIGN = None


def set_board_size(n):
    '''
    Hopefully nobody tries to run both 9x9 and 19x19 game instances at once.
    Also, never do "from go import N, W, ALL_COORDS, EMPTY_BOARD".
    '''
    global N, N2, ALL_COORDS, EMPTY_BOARD, NEIGHBORS, DIAGONALS, COLLUM_STR, PASS, RESIGN
    if N == n: return
    N = n
    N2 = n*n
    PASS = N2
    RESIGN = N2+1
    ALL_COORDS = [i for i in range(N2)]
    EMPTY_BOARD = [0 for i in range(N2)]
    def check_bounds(c):
        if c<0 or c>=N2:
            return False
        x,y=unflatten_coords(c)
        return x>=0 and x<N and y>=0 and y<N
    for i in ALL_COORDS:
        x1,y1=unflatten_coords(i)
        ni = []
        di = []
        for (x,y) in [(x1+1,y1),(x1-1,y1),(x1,y1+1),(x1,y1-1)]:
            if x>=0 and x<N and y>=0 and y<N:
                #{i: list(filter(check_bounds, [i+1, i-1, i+n, i-n])) for i in ALL_COORDS}
                ni.append(y*n+x)
        for (x,y) in [(x1+1,y1+1),(x1-1,y1+1),(x1+1,y1-1),(x1-1,y1-1)]:
            if x>=0 and x<N and y>=0 and y<N:
                #{i: list(filter(check_bounds, [i+n+1, i-n+1, i+n-1, i-n-1])) for i in ALL_COORDS}
                di.append(y*n+x)
        NEIGHBORS[i] = ni
        DIAGONALS[i] = di

    COLLUM_STR='ABCDEFGHJKLMNOPQRSTUVWXYZ'

class AnalyseData(namedtuple('AnalyseData', ['move', 'allvisits', 'visits', 'winrate','nextmoves'])): pass

def result_str(result):
    result = result.upper()
    winner = "黑" if result[0]=="B" else "白"
    winp = ""
    if result[2:]=="RESIGN":
        winp = "中盘胜"
    elif result[2:]=="TIME":
        winp = "对方超时胜"
    else:
        p = float(result[2:])
        winp = "胜{}目".format(p)
    return winner+winp

def get_point_str(p):
    return "{}领先{:.1f}目".format(get_color_str(p), abs(p))

def get_points_str(p, c):
    return get_point_str(p if c==BLACK else -p)

def get_color_str(color):
    if color not in [BLACK,WHITE]:
        return '黑' if color>1 else '白'
    return '黑' if color==BLACK else '白'

def get_legal_str(ind):
    ills = ['合法', '有子', '打劫(全局同形)', '自杀', '填眼', '不该PASS', '该PASS']
    return ills[ind]

def oppo_color(color):
    return BLACK+WHITE - color

def get_coor_from_gtp(cmd):
    if not cmd:
        return None
    ucmd = cmd.upper()
    if ucmd=="PASS":
        return PASS
    if ucmd=="RESIGN":
        return RESIGN
    if ucmd=="UNDO":
        return UNDO
    if len(ucmd)<2 or len(ucmd)>3 or ucmd[0] not in COLLUM_STR:
        return None
    x = COLLUM_STR.index(ucmd[0])
    y = N-int(ucmd[1:])
    return y*N+x

def get_cmd_from_coor(coor):
    if coor is None or coor == PASS:
        return 'PASS'
    elif coor==RESIGN:
        return 'RESIGN'
    elif coor == UNDO:
        return 'UNDO'
    elif coor>=0 and coor<N2:
        y, x = divmod(coor, N)
        return COLLUM_STR[x]+str(N-y)
    return 'PASS'

def flatten_coords(c):
    if not c:
        c=PASS
    r = N * c[1] + c[0]
    if r>N*N+1:
        return N*N
    return r

def unflatten_coords(f):
    if f<0 or f>N*N+2:
        return None
    return (f%N,f//N)

def take_n(n, iterable):
    return list(itertools.islice(iterable, n))

'''
def get_coor_from_vertex(vertex):
    if vertex == gtp.PASS:
        return PASS
    elif vertex == gtp.RESIGN:
        return RESIGN
    return (N - vertex[1], vertex[0]-1)

def get_vertex_from_coor(coor):
    if coor is None or coor==PASS:
        return gtp.PASS
    elif coor==RESIGN:
        return gtp.RESIGN
    return (coor[1]+1, N-coor[0])

def get_cmd_from_vertex(vertex):
    if vertex[0] in range(1, N+1) and vertex[1] in range(1, N+1):
        return COLLUM_STR[vertex[0]-1]+str(vertex[1])
    elif vertex == gtp.RESIGN:
        return 'RESIGN'
    return 'PASS'
'''

def place_stones(bd, color, stones):
    #assert stones is list, '检查stones, 应该是个list数组：' + str(type(stones))
    for s in stones:
        if s>=0 and s<N2:
            bd[s] = color

def fill_stone(position):
    pos = copy.deepcopy(position)
    move = None
    moves = None
    while not pos.is_gameover:
        if move is None:
            moves = pos.get_can_moves()
        if not moves:
            move = PASS
        else:
            move = moves[0]
        ill, caps = pos.play_move(move, check_legal=True)
        if ill==0:
            move = None
        else:
            moves.remove(move)
    return pos

def simulate_game(position):
    pos = copy.deepcopy(position)
    move = None
    moves = None
    while not pos.is_gameover:
        if move is None:
            moves = pos.get_can_moves()
        if not moves:
            move = PASS
        else:
            move = random.choice(moves)
        ill, caps = pos.play_move(move, check_legal=True)
        if ill==0:
            move = None
        else:
            moves.remove(move)
    return pos

def delete_dead(position):
    pos = simulate_game(position)
    #changed = np.where(position.stones!=pos.stones)
    #pos = position.copy()
    #place_stones(pos.stones, EMPTY, dead)
    return pos

set_board_size(19)
