#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

from collections import namedtuple
import copy
import math
import numpy as np
import go
import dbhash
import random
import sgfparser
import config
#import group

MISSING_GROUP_ID = -1
GROUP_CHAIN_END = -2

class PlayerMove(namedtuple('PlayerMove', ['move', 'color', 'stones', 'hash'])): pass
    #def __init__(self, move, color, captured=None, ko=None, score=0):
    #    self.move=move
    #    self.color = color
    #    self.captured=captured
    #    self.ko = ko
    #    self.score = score

def find_reached(board, c, rc=None):
    color = board[c]
    chain = set([c])
    reached = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in go.NEIGHBORS[current]:
            if board[n] == color and not n in chain:
                frontier.append(n)
            elif (rc and board[n]==rc) or (rc is None and board[n] != color):
                reached.add(n)
    return chain, reached

def is_eye(bd, c):
    'Check if c is surrounded on all sides by 1 color, and return that color'
    if bd[c] != go.EMPTY: return None
    neighbors = {bd[n] for n in go.NEIGHBORS[c]}
    if len(neighbors) == 1 and not go.EMPTY in neighbors:
        return list(neighbors)[0]
    else:
        return None

def is_trueeye(board, c):
    'Check if c is an eye, for the purpose of restricting MC rollouts.'
    color = is_eye(board, c)
    if color is None:
        return None
    diagonals = go.DIAGONALS[c]
    dnl = len(diagonals)
    diagonal_faults = min(dnl, 3)
    for d in diagonals:
        if board[d] == color:
            diagonal_faults -= 1
        elif is_eye(board, d) == color:
            diagonal_faults -= 1
    if diagonal_faults > 0:
        return None
    else:
        return color

class Board():
    def __init__(self, stones=None, n=0, komi=7.5, caps=(0, 0), ko=None, 
                 to_move=go.BLACK, handicap=0, hash=None, ko_hash=None):
        '''
        board: a numpy array
        n: an int representing moves played so far
        komi: a float, representing points given to the second player.
        caps: a (int, int) tuple of captures for B, W.
        lib_tracker: a LibertyTracker object
        ko: a Move
        recent: a tuple of PlayerMoves, such that recent[-1] is the last move.
        to_move: go.BLACK or go.WHITE
        '''
        self.stones = np.zeros(go.N2, dtype = np.int8) if stones is None else stones
        self.clear_group()
        self.recent = []
        self.step = n
        self.komi = komi
        self.caps = caps
        self.ko = ko
        self.first_color = to_move if n%2==0 else go.oppo_color(to_move)
        self.to_move = to_move
        self.player1_name = 'ZiGo1'
        self.player2_name = 'ZiGo2'
        self.points = 0
        self.result = None
        self.handicap = handicap if handicap is not None else 0
        if handicap is not None and handicap>0:
            self.set_handicap(handicap)
        self.size = go.N
        self.hash = dbhash.get_hash(self) if hash is None else hash
        self.ko_hash = dbhash.get_ko_hash(self) if ko_hash is None else ko_hash

    def set_handicap(self, handicap):
        handicap = min(9, handicap)
        hinds = [go.flatten_coords(p) for p in [(3,3),(15,15),(3,15),(15,3),(3,9),(15,9),(9,3),(15,3),(9,9)]]
        self.stones = np.zeros(go.N2, dtype = np.int8)
        self.handicap = handicap
        for i in range(handicap):
            self.stones[hinds[i]] = go.BLACK

    def clear(self):
        self.__init__(komi=self.komi)

    def clear_group(self):
        self.group_stones = np.array([MISSING_GROUP_ID for i in range(go.N2)])
        self.group_ids = np.array([MISSING_GROUP_ID for i in range(go.N2)])
        self.group_libs = np.array([MISSING_GROUP_ID for i in range(go.N2)])

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memodict={}):
        bd = Board(self.stones.copy(), self.step, self.komi, self.caps, self.ko, 
                     self.to_move, self.handicap, self.hash, self.ko_hash)
        bd.group_libs = copy.deepcopy(self.group_libs)
        bd.group_stones = copy.deepcopy(self.group_stones)
        bd.group_ids = copy.deepcopy(self.group_ids)
        bd.recent = copy.deepcopy(self.recent)
        return bd

    def __str__(self):
        return 'step: {}, to_move: {}, hash: {}, result: {}, komi: {}, handicap: {}'.format(
                self.step, self.to_move, self.hash, self.result, self.komi, self.handicap)

    def is_move_suicidal(self, move):
        potential_libs = 0
        grps = set()
        for n in go.NEIGHBORS[move]:
            if self.stones[n] == go.EMPTY:
                # 至少有一气，所以不是自杀
                return False
            if self.group_libs[n]==1 and self.stones[n] != self.to_move:
                # 敌方棋子只有一气，可以杀死
                return False
            if self.stones[n] == self.to_move:
                grps.add(self.group_ids[n])
        for g in grps:
            potential_libs += self.group_libs[g]-1
        # 可能周围的友方棋都只有一气，这样就是自杀
        return potential_libs<=0

    def is_move_legal(self, move):
        '''
        1-有子，2-打劫，3-自杀，4-填眼
        '''
        if move is None or move == go.PASS or move == go.RESIGN:
            return 0
        if self.stones[move] != go.EMPTY:
            return 1
        if self.is_move_suicidal(move):
            return 3
        if is_trueeye(self.stones, move) == self.to_move:
            return 4
        if self.ko is None:
            return 0
        tb = self.try_move(move)
        his = 8
        if self.step>go.N2:
            his=16
        pl = min(len(self.recent), his)
        for i in range(1, pl):
            if tb.ko_hash == self.recent[-i].hash:
                return 2
        return 0

    @property
    def is_gameover(self):
        if len(self.recent)<3:
            return False
        if self.recent[-1].move == go.RESIGN or \
            (self.recent[-1].move==go.PASS and \
             self.recent[-2].move==go.PASS):
            return True
        return False

    def flip_playerturn(self, mutate=False):
        pos = self if mutate else copy.deepcopy(self)
        pos.ko = None
        pos.to_move  = go.oppo_color(self.to_move)
        return pos

    def try_move(self, c):
        pos = copy.deepcopy(self)
        pos.play_move(c, check_legal=False)
        return pos

    def play_move(self, move, color=None, check_legal=True):
        if check_legal:
            ill=self.is_move_legal(move)
            if ill>0:
                config.logger.debug('{}棋着法{} {}，不合法！'.format(go.get_color_str(self.to_move), \
                    go.get_cmd_from_coor(move), go.get_legal_str(ill)))
                return ill, None
        if color is None:
            color = self.to_move
        self.recent.append(PlayerMove(move, self.to_move, self.stones.copy(), self.ko_hash))

        if self.ko:
            self.hash = self.hash ^ dbhash.ZOBRIST[go.KO][dbhash.get_index(self.ko)]
        self.hash = self.hash ^ dbhash.ZOBRIST_BLACKTOMOVE
        if move is None or move == go.PASS or move==go.RESIGN:  #vertex(20,19) coor(19, 0)
            self.step += 1
            self.to_move = go.oppo_color(self.to_move)
            self.ko = None
            return 0, None

        potential_ko = is_eye(self.stones, move)

        opp_color = go.oppo_color(color)
        move_ind = dbhash.get_index(move)
        self.hash = self.hash ^ dbhash.ZOBRIST[go.EMPTY][move_ind]
        self.ko_hash = self.ko_hash ^ dbhash.ZOBRIST[go.EMPTY][move_ind]
        self.stones[move] = color
        self.hash = self.hash ^ dbhash.ZOBRIST[color][move_ind]
        self.ko_hash = self.ko_hash ^ dbhash.ZOBRIST[color][move_ind]
        captured_stones = self.add_stone(move, color)
        go.place_stones(self.stones, go.EMPTY, captured_stones)
        if captured_stones:
            for s in captured_stones:
                ind = dbhash.get_index(s)
                self.hash = self.hash ^ dbhash.ZOBRIST[opp_color][ind]
                self.hash = self.hash ^ dbhash.ZOBRIST[go.EMPTY][ind]
                self.ko_hash = self.ko_hash ^ dbhash.ZOBRIST[opp_color][ind]
                self.ko_hash = self.ko_hash ^ dbhash.ZOBRIST[go.EMPTY][ind]


        if len(captured_stones) == 1 and potential_ko == opp_color:
            new_ko = list(captured_stones)[0]
        else:
            new_ko = None
        if new_ko:
            self.hash = self.hash ^ dbhash.ZOBRIST[go.KO][dbhash.get_index(new_ko)]

        if self.to_move == go.BLACK:
            new_caps = (self.caps[0] + len(captured_stones), self.caps[1])
        else:
            new_caps = (self.caps[0], self.caps[1] + len(captured_stones))

        self.step += 1
        self.caps = new_caps
        self.ko = new_ko
        self.to_move = go.oppo_color(self.to_move)
        return 0, captured_stones

    def replay_move(self, move, color=None):
        if move is None or move == go.PASS or move == go.RESIGN:  #vertex(20,19) coor(19, 0)
            self.play_move(go.PASS)
            return
        if color is None:
            color = self.to_move
        self.recent.append(PlayerMove(move, self.to_move, self.stones.copy(), self.ko_hash))
        self.stones[move] = color
        captured_stones = self.add_stone(move, color)
        go.place_stones(self.stones, go.EMPTY, captured_stones)
        if self.to_move == go.BLACK:
            new_caps = (self.caps[0] + len(captured_stones), self.caps[1])
        else:
            new_caps = (self.caps[0], self.caps[1] + len(captured_stones))
        self.caps = new_caps
        self.step += 1
        self.to_move = go.oppo_color(self.to_move)

    def undo(self):
        if len(self.recent)<1:
            return
        self.stones = self.recent[-1].stones.copy()
        del self.recent[-1]
        self.step -= 1
        self.to_move = go.oppo_color(self.to_move)
        self.build_groups()

    def get_board(self, step):
        if step == self.step:
            return self.copy()
        if step > self.step or step<0:
            return None
        board = Board(komi = self.komi, to_move=self.first_color, handicap=self.handicap)
        for i, pm in enumerate(self.recent):
            if i< step:
                board.play_move(pm.move, pm.color, False)
        return board
        board = copy.deepcopy(self)
        board.stones = board.recent[step].stones.copy()
        if step < 1:
            board.recent = []
        else:
            board.recent = board.recent[0:step]
        board.to_move = board.get_color(step)
        board.step = step
        if board.first_color == go.BLACK:
            cb = 1
            cw = 0
        else:
            cb = 0
            cw = 1
        board.caps = (step//2-np.count_nonzero(board.stones[board.stones==go.WHITE])+cw,
                      step//2-np.count_nonzero(board.stones[board.stones==go.BLACK])+cb)
        board.build_groups()
        return board

    def get_try_stones(self, move):
        if move==go.PASS:
            return self.stones.copy()
        pos = self.copy()
        captured_stones = pos.add_stone(move, pos.to_move)
        stones = pos.stones
        go.place_stones(stones, go.EMPTY, captured_stones)
        stones[move] = self.to_move
        return stones

    def get_past_stones(self, n):
        return self.get_stones(self.step-n)

    def get_stones(self, step):
        if step<1 and len(self.recent)>0:
            return self.recent[0].stones.copy()
        elif step == self.step or len(self.recent)<1:
            return self.stones.copy()
        return self.recent[step].stones.copy()

    def from_history(self, history, step):
        self.clear()
        self.first_color = history[0].color
        self.to_move = history[step].color
        self.step = step
        self.stones = history[step].stones.copy()
        self.recent = history[0:step]
        self.build_group()

    def from_moves(self, moves, result, first_color=go.BLACK):
        self.clear()
        self.first_color=first_color
        self.to_move=first_color
        self.result = result
        for move in moves:
            self.replay_move(move)

    def get_moves(self):
        return [s for s in go.ALL_COORDS if self.is_move_legal(s)==0]

    def get_can_moves(self):
        return [s for s in go.ALL_COORDS if self.stones[s]==go.EMPTY]

    def get_color(self, step):
        return self.first_color if step%2==0 else go.oppo_color(self.first_color)
        #if step>len(self.recent) or step<0:
        #    return go.BLACK
        #return self.recent[step].color

    def get_step_from_move(self, move):
        for i,rc in enumerate(self.recent):
            if rc.move==move:
                return i
        return 0
        
    def scorebw(self):
        'Return score from B perspective. If W is winning, score is negative.'
        working_board = self.stones.copy()
        bs = []
        ws = []
        has_uk = False
        while go.EMPTY in working_board:
            bk = []
            wk = []
            uk = []
            unassigned_spaces = np.where(working_board == go.EMPTY)
            c = unassigned_spaces[0][0]
            territory, borders = find_reached(working_board, c)
            border_colors = set(working_board[b] for b in borders)
            X_border = go.BLACK in border_colors
            O_border = go.WHITE in border_colors
            if X_border and not O_border:
                bk += territory
            elif O_border and not X_border:
                wk += territory
            else:
                uk += territory
                has_uk = True
            go.place_stones(working_board, go.BLACK, bk)
            go.place_stones(working_board, go.WHITE, wk)
            go.place_stones(working_board, go.UNKNOWN, uk)
            bs += bk
            ws += wk
        blacks = np.count_nonzero(working_board == go.BLACK)
        whites = np.count_nonzero(working_board == go.WHITE)
        return (blacks - whites - self.komi - self.handicap, bs, ws, has_uk)

    def score(self):
        score, bs, ws, has_uk = self.scorebw()
        return score

    def get_result(self):
        return sgfparser.unparse_result(self.points)

    def final_score(self):    
        score, bs, ws, has_uk = self.scorebw()
        if has_uk:
            pos = go.simulate_game(self)
            score = pos.score()
        return score

    def build_groups(self):
        self.clear_group()
        for p in go.ALL_COORDS:
            if self.stones[p]!=go.EMPTY and self.group_ids[p]==MISSING_GROUP_ID:
                c = self.stones[p]
                sts, libs = find_reached(self.stones, p, go.EMPTY)
                lib = len(libs)
                for s in sts:
                    self.group_ids[s] = p
                    self.group_libs[s] = lib
                sts.remove(p)
                i = p
                for s in sts:
                    self.group_stones[i] = s       
                    i = s
                self.group_stones[i] = GROUP_CHAIN_END

    def add_stone(self, move, color):
        captured_stones = []
        opp_ids = set()
        friendly_ids = set()
        self.group_libs[move] = 0
        for n in go.NEIGHBORS[move]:
            if self.stones[n] == go.EMPTY:
                self.group_libs[move] += 1
            else:
                gid = self.group_ids[n]
                if self.stones[gid] == color:
                    friendly_ids.add(gid)
                else:
                    opp_ids.add(gid)

        self.group_ids[move] = move
        self.group_stones[move] = GROUP_CHAIN_END
        new_group = move
        for group in friendly_ids:
            new_group = self.merge_groups(group, new_group)

        for group in opp_ids:
            if self.group_libs[group] == 1:
                captured = self.handle_captures(group)
                captured_stones += captured
            else:
                self.add_libs(group, -1)

        # suicide is illegal
        if self.group_libs[new_group] == 0:
            config.logger.error("Move at {} would commit suicide!\n".format(go.get_cmd_from_coor(move)))
        return captured_stones

    def merge_groups(self, group1, group2):
        if group2<group1:
            t = group1
            group1=group2
            group2 = t
        libs = set()
        linked = False
        s = group1
        while s>=0:
            for n in go.NEIGHBORS[s]:
                if self.stones[n] == go.EMPTY:
                    libs.add(n)
            s1 = self.group_stones[s]
            if not linked and s1==GROUP_CHAIN_END:
                self.group_stones[s] = group2
                s = group2
                linked = True
            else:
                s = s1
        s = group1
        lib = len(libs)
        while s>=0:
            self.group_ids[s] = group1
            self.group_libs[s] = lib
            s = self.group_stones[s]
        return group1

    def add_libs(self, group, libs):
        s = group
        while s>=0:
            self.group_libs[s] += libs
            s = self.group_stones[s]

    def handle_captures(self, group):
        i=group;
        caps = []
        while i>=0:
            s = i
            caps.append(s);
            i = self.group_stones[s];
            self.group_stones[s] = MISSING_GROUP_ID;
            self.group_ids[s] = MISSING_GROUP_ID;
            self.group_libs[s] = MISSING_GROUP_ID;
            added = set()
            for n in go.NEIGHBORS[s]:
                grpid = self.group_ids[n];
                if grpid >= 0 and grpid!=group:
                    added.add(grpid);
            for id in added:
                self.add_libs(id, 1)
        return caps;

