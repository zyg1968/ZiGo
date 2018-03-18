#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

from collections import namedtuple
#import threading
import sys
import go
import dbhash
import config
import strategies
import dataset

sys.setrecursionlimit(100000) #例如这里设置为一百万

class TrainData(namedtuple('TrainData', ['board', 'points'])): pass

class AllSearch:
    def __init__(self, board):
        self.root=None
        self.rootboard = board
        self.data = []
        self.sql = dbhash.SqliteDb()
        self.datas = dataset.DataSet()

    def start(self):
        print ("开始全搜索……", file=sys.stderr)
        if not self.root:
            self.root = FastNode(None, go.PASS, 0.0)
            self.expand(self.root, self.rootboard)
        self.search_tree(self.root, self.rootboard, 0)
        if len(self.data)>0:
            print("保存到数据库, self.data len:", len(self.data), file=sys.stderr)
            self.sql.save(self.data)
            self.data = []
        self.sql.close()
        if self.datas.data_size>1600:
            self.datas.save()
            self.datas.clear()
        print ("退出全搜索……", file=sys.stderr)

    def search_tree(self, node, board, depth=0):
        depth += 1
        if len(self.data)>1000:
            print("保存到数据库，depth:", depth, ", self.data len:", len(self.data), file=sys.stderr)
            self.sql.save(self.data)
            self.data = []
        
        if node.search_over:
            return node.points/(node.visits if node.visits>0 else 1)
        if depth>go.N*go.N*2:
            node.search_over = True
            return 0
        p = self.sql.search(board)
        if p>-9999:
            node.update(p)
            return p;

        move_values = [0.0 for i in range(go.N*go.N+2)]
        tomove = board.to_move
        if node.expanded and (board.is_gameover or not node.childs):
            node.search_over = True
            score = strategies.fast_score(board)
            ps = score if tomove==go.BLACK else -score
            node.update(ps)
            self.data.append(TrainData(board, ps))
            move_values[-2] = 1.0
            self.datas.add_from_node(board, move_values, ps)
            if self.datas.data_size>6400:
                self.datas.save()
                self.datas.clear()
            return ps

        node.points=0.0
        for child in node.childs:
            if not config.running:
                break
            b = board.copy()
            b.play_move(child.move)
            v = 0.0
            if not child.expanded:
                self.expand(child, b)
                v = self.search_tree(child, b, depth)
            elif not child.search_over:
                v = self.search_tree(child, b. depth)
            ind = go.flatten_coords(child.move)
            move_values[ind] = v
        visits = node.visits if node.visits>0 else 1
        #maxnode = max(node.childs, key=lambda x:x.points)
        #move_values[go.flatten_coords(maxnode.move)] = 1.0
        if config.running and depth<int(go.N*go.N*2):
            node.search_over = True
            self.data.append(TrainData(board, node.points/visits))
            self.datas.add_from_node(board, move_values, node.points/visits)
            if self.datas.data_size>6400:
                self.datas.save()
                #print("保存训练数据，depth:", depth, ", self.datas len:", self.datas.data_size)
                self.datas.clear()
        return node.points/visits

    def expand(self, node, b):
        moves = b.get_moves()
        for m in moves:
            node.add_child(m, 0)
        if not moves:
            node.add_child(go.PASS, 0)
        #node.add_child(go.RESIGN, 0)
        #print("Node expanded: ", len(moves), file=sys.stderr)
        node.expanded=True


class FastNode:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, parent, move=None, points=0):
        self.parent = parent
        self.move = move  # the move that got us to this node - "None" for the root node
        self.childs = []
        self.points = points
        self.visits = 0
        self.expanded = False
        self.search_over = False

    def add_child(self, move, points):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = FastNode(self, move=move, points=points)
        self.childs.append(n)
        return n

    def update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.points += result
        if self.parent:
            self.parent.update(-result)

    def copy(self):
        r = Node(None, self.move, self.points)
        r.childs=self.childs[:]
        r.visits=self.visits
        return r

    def select(self):
        for child in childs:
            if not child.search_over:
                return child
        self.search_over = True
        return None

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.points) + "/" + str(self.visits) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childs:
            s += str(c) + "\n"
        return s
