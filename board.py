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

class PlayerMove(namedtuple('PlayerMove', ['move', 'color', 'stones', 'hash'])): pass
    #def __init__(self, move, color, captured=None, ko=None, score=0):
    #    self.move=move
    #    self.color = color
    #    self.captured=captured
    #    self.ko = ko
    #    self.score = score

def place_stones(board, color, stones):
    #assert stones is list, '检查stones, 应该是个list数组：' + str(type(stones))
    for s in stones:
        if s[0]>=0 and s[0]<go.N and s[1]>=0 and s[1]<go.N:
            board[s] = color

def is_group_dead(group, board, scores):
    eye=set()
    for s in group.liberties:
        if is_eyeish(board, s) and not eye.intersection(set(go.NEIGHBORS(s))):
            eye.add(s)
    return len(eye)<2 and len(scores)<3

def get_group_scores(group, board):
    scores = set()
    for s in group.liberties:
        v = 0.0
        territory, borders = find_reached(board, s)
        for b in borders:
            cl = board[b]
            dist = math.hypot(b[0]-s[0], b[1]-s[1])
            v += cl/(2.0**dist)
        if v*group.color>0.4:
            scores.add(s)
    return scores

def find_reached(board, c):
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
            elif board[n] != color:
                reached.add(n)
    return chain, reached

def is_koish(board, c):
    'Check if c is surrounded on all sides by 1 color, and return that color'
    if board[c] != go.EMPTY: return None
    neighbors = {board[n] for n in go.NEIGHBORS[c]}
    if len(neighbors) == 1 and not go.EMPTY in neighbors:
        return list(neighbors)[0]
    else:
        return None

def is_eyeish(board, c):
    'Check if c is an eye, for the purpose of restricting MC rollouts.'
    color = is_koish(board, c)
    if color is None:
        return None
    diagonals = go.DIAGONALS[c]
    dnl = len(diagonals)
    diagonal_faults = min(dnl, 3)
    for d in diagonals:
        if board[d] == color:
            diagonal_faults -= 1
        elif is_koish(board, d) == color:
            diagonal_faults -= 1
    if diagonal_faults > 0:
        return None
    else:
        return color

class Group(namedtuple('Group', ['stones', 'liberties', 'color'])):
    def __eq__(self, other):
        return self.stones == other.stones and self.liberties == other.liberties and self.color == other.color


class Board():
    def __init__(self, stones=None, n=0, komi=7.5, caps=(0, 0), groups=None, ko=None, 
                 recent=None, to_move=go.BLACK, hash=None, ko_hash=None):
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
        self.stones = np.zeros([go.N, go.N], dtype=np.int8) if stones is None else stones
        self.step = n
        self.komi = komi
        self.caps = caps
        self.ko = ko
        self.groups = groups or self.build_groups()
        self.recent = [] if recent is None else recent
        self.first_color = to_move if n%2==0 else -to_move
        self.to_move = to_move
        self.player1_name = 'ZiGo1'
        self.player2_name = 'ZiGo2'
        self.points = 0
        self.result = 0
        self.handicap = 0
        self.size = go.N
        self.hash = dbhash.get_hash(self) if hash is None else hash
        self.ko_hash = dbhash.get_ko_hash(self) if ko_hash is None else ko_hash

    def clear(self):
        self.__init__(komi=self.komi)

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memodict={}):
        stones = np.copy(self.stones)
        groups = copy.deepcopy(self.groups)
        recent = copy.deepcopy(self.recent)
        return Board(stones, self.step, self.komi, self.caps, groups, 
                     self.ko, recent, self.to_move, self.hash, self.ko_hash)

    def __str__(self):
        return 'step: {}, to_move: {}, hash: {}, groups: {}, result: {}, komi: {}, handicap: {}'.format(
                self.step, self.to_move, self.hash, len(self.groups), self.result, self.komi, self.handicap)

    def is_move_suicidal(self, move):
        potential_libs = set()
        for n in go.NEIGHBORS[move]:
            neighbor_group = self.get_group(n)
            if not neighbor_group:
                # 至少有一气，所以不是自杀
                return False
            if neighbor_group.color == self.to_move:
                potential_libs |= neighbor_group.liberties
            elif len(neighbor_group.liberties) == 1:
                # 敌方棋子只有一气，可以杀死
                return False
        # 可能周围的友方棋都只有一气，这样就是自杀
        potential_libs -= set([move])
        return not potential_libs

    def is_move_legal(self, move):
        '''
        1-有子，2-打劫，3-自杀，4-填眼
        '''
        if move is None or move == go.PASS:
            return 0
        if self.stones[move] != go.EMPTY:
            return 1
        if self.is_move_suicidal(move):
            return 3
        if is_eyeish(self.stones, move) == self.to_move:
            return 4
        tb = self.try_move(move)
        his = 8
        if self.step>go.N*go.N:
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
                #print('%s棋着法%s %s，不合法！' % (get_color_str(self.to_move), get_cmd_from_coor(move), get_legal_str(ill)))
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

        potential_ko = is_koish(self.stones, move)

        opp_color = go.oppo_color(color)
        move_ind = dbhash.get_index(move)
        self.hash = self.hash ^ dbhash.ZOBRIST[go.EMPTY][move_ind]
        self.ko_hash = self.ko_hash ^ dbhash.ZOBRIST[go.EMPTY][move_ind]
        self.stones[move] = color
        self.hash = self.hash ^ dbhash.ZOBRIST[color][move_ind]
        self.ko_hash = self.ko_hash ^ dbhash.ZOBRIST[color][move_ind]
        captured_stones = self.add_stone(color, move)
        place_stones(self.stones, go.EMPTY, captured_stones)
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
        if move is None or move == go.PASS:  #vertex(20,19) coor(19, 0)
            self.play_move(go.PASS)
            return
        if color is None:
            color = self.to_move
        self.recent.append(PlayerMove(move, self.to_move, self.stones.copy(), self.ko_hash))
        self.stones[move] = color
        captured_stones = self.add_stone(color, move)
        place_stones(self.stones, go.EMPTY, captured_stones)
        self.step += 1
        self.to_move = go.oppo_color(self.to_move)

    def undo(self):
        if len(self.recent)<1:
            return
        self.stones = np.copy(self.recent[-1].stones)
        del self.recent[-1]
        self.step -= 1
        self.to_move = go.oppo_color(self.to_move)
        self.groups=self.build_groups()

    def get_board(self, step):
        if step == self.step:
            return self.copy()
        if step > self.step or step<0:
            return None
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
        return board

    def get_try_stones(self, move):
        if move==go.PASS:
            return self.stones.copy()
        pos = self.copy()
        captured_stones = pos.add_stone(pos.to_move, move)
        stones = pos.stones.copy()
        place_stones(stones, go.EMPTY, captured_stones)
        stones[move] = self.to_move
        return stones

    def get_past_stones(self, n):
        return self.get_stones(self.step-n)

    def get_stones(self, step):
        if step<1 and len(self.recent)>0:
            return self.recent[0].stones
        elif step == self.step or len(self.recent)<1:
            return self.stones
        return self.recent[step].stones

    def from_history(self, history, step):
        self.clear()
        self.first_color = history[0].color
        self.to_move = history[step].color
        self.step = step
        self.stones = history[step].stones
        self.recent = history[0:step]

    def from_moves(self, moves, result, first_color=go.BLACK):
        self.clear()
        self.first_color=first_color
        self.to_move=first_color
        self.result = result
        for move in moves:
            self.replay_move(move)

    def get_moves(self):
        return [s for s in go.ALL_COORDS if self.is_move_legal(s)==0]

    def get_color(self, step):
        return self.first_color if step%2==0 else go.oppo_color(self.first_color)
        
    def score(self):
        'Return score from B perspective. If W is winning, score is negative.'
        working_board = np.copy(self.stones)
        while go.EMPTY in working_board:
            bk = []
            wk = []
            uk = []
            unassigned_spaces = np.where(working_board == go.EMPTY)
            c = unassigned_spaces[0][0], unassigned_spaces[1][0]
            territory, borders = find_reached(working_board, c)
            border_colors = set(working_board[b] for b in borders)
            X_border = go.BLACK in border_colors
            O_border = go.WHITE in border_colors
            if X_border and not O_border:
                bk += territory
            elif O_border and not X_border:
                wk += territory
            else:
                for k in territory:
                    v = 0.0
                    for b in borders:
                        cl = 1 if working_board[b]==go.BLACK else -1
                        dist = math.hypot(b[0]-k[0], b[1]-k[1])
                        v += cl/(2.0**dist)
                    if v>0.5:
                        bk.append(k)
                    elif v<-0.5:
                        wk.append(k)
                    else:
                        uk.append(k)
            place_stones(working_board, go.BLACK, bk)
            place_stones(working_board, go.WHITE, wk)
            place_stones(working_board, go.UNKNOWN, uk)
        return np.count_nonzero(working_board == go.BLACK) - np.count_nonzero(working_board == go.WHITE) - self.komi

    def build_groups(self):
        board = np.copy(self.stones)
        self.groups = []
        for color in (go.WHITE, go.BLACK):
            while color in board:
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                chain, reached = find_reached(board, coord)
                liberties = set(r for r in reached if board[r] == go.EMPTY)
                new_group = Group(chain, liberties, color)
                self.groups.append(new_group)
                place_stones(board, go.FILL, chain)
        return self.groups

    def add_stone(self, color, c):
        captured_stones = set()
        opponent_neighboring_groups = []
        friendly_neighboring_groups = []
        empty_neighbors = set()

        for n in go.NEIGHBORS[c]:
            neighbor_group = self.get_group(n)
            if neighbor_group:
                is_in = False
                if neighbor_group.color == color:
                    for g in friendly_neighboring_groups:
                        if g==neighbor_group:
                            is_in=True
                            break
                    if not is_in:
                        friendly_neighboring_groups.append(neighbor_group)
                else:
                    for g in opponent_neighboring_groups:
                        if g==neighbor_group:
                            is_in=True
                            break
                    if not is_in:
                        opponent_neighboring_groups.append(neighbor_group)
            else:
                empty_neighbors.add(n)

        new_group = self.create_group(color, c, empty_neighbors)
        for group in friendly_neighboring_groups:
            new_group = self.merge_groups(group, new_group)

        for group in opponent_neighboring_groups:
            if len(group.liberties) == 1:
                captured = self.capture_group(group)
                captured_stones.update(captured)
            else:
                self.update_liberties(group, remove={c})
        self.handle_captures(captured_stones)

        # suicide is illegal
        if len(new_group.liberties) == 0:
            raise IllegalMove("Move at {} would commit suicide!\n".format(c))
        return captured_stones

    def create_group(self, color, c, liberties):
        new_group = Group(set([c]), liberties, color)
        self.groups.append(new_group)
        return new_group

    def get_group(self, c):
        if self.groups:
            for g in self.groups:
                if c in g.stones:
                    return g
        return None

    def merge_groups(self, group1, group2):
        group1.stones.update(group2.stones)
        self.update_liberties(group1, add=group2.liberties, remove=(group2.stones | group1.stones))
        self.groups.remove(group2)
        return group1

    def capture_group(self, group):
        stones = group.stones
        self.groups.remove(group)
        return stones

    def update_liberties(self, group, add=None, remove=None):
        if add:
            group.liberties.update(add)
        if remove:
            group.liberties.difference_update(remove)

    def handle_captures(self, captured_stones):
        for s in captured_stones:
            for n in go.NEIGHBORS[s]:
                group = self.get_group(n)
                if group:
                    self.update_liberties(group, add={s})




