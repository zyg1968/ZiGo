import copy
import math
import random
import sys
import time

import gtp
import numpy as np

import go
import utils
from functools import reduce
import config

MAX_NODES = 5000

# Draw moves from policy net until this threshold, then play moves randomly.
POLICY_CUTOFF_DEPTH = int(go.N * go.N * 1.0) # 270 moves for a 19x19
# This speeds up the simulation, and it also provides a logical cutoff
# for which moves to include for reinforcement learning.
# However, some situations end up as "dead, but only with correct play".
POLICY_FINISH_MOVES = int(go.N * go.N * 0.2) # 72 moves for a 19x19
# Random play can destroy the subtlety of these situations, so we'll play out
# a bunch more moves from a smart network before playing out random moves.

def get_noise(vs):
    if not vs or len(vs)<3:
        return vs
    n2 = len(vs)
    dirch = np.random.dirichlet([0.03 * (n2 - 2) / (n2)] * n2)
    r = []
    for v, d in zip(vs, dirch):
        r.append(v * 0.75 + d * 0.25 * go.N*go.N)
    return r

def get_values(node):
    vs = [0 for i in range(go.N*go.N+2)]
    for child in node.childs:
        m = utils.flatten_coords(child.move)
        vs[m] = child.V
    return vs

def pass_or_resign(node):
    vs = get_values(node)
    if not node.childs:
        vs[go.N*go.N] = go.N*go.N
        return go.PASS, vs, None
    maxc = max(node.childs, key=lambda c:c.V)
    if maxc.V == 0:
        vs[go.N*go.N] = go.N*go.N
        return go.PASS, vs,maxc
    if maxc.points+maxc.V*10<0:
        vs[go.N*go.N+1] = go.N*go.N
        return go.RESIGN, vs, maxc
    return 0, vs, maxc

def get_node_num(node, n=0):
    n += len(node.childs)
    for child in node.childs:
        if child.childs:
            n += get_node_num(child, n)
    return n

def show_node_tree(node, depth=0):
    if depth>10 or not node.childs:
        return
    depth += 1
    child = max(node.childs, key=lambda x: x.V)
    print("{}->".format(go.get_cmd_from_coor(child.move)), end="")
    show_node_tree(child, depth)


def sorted_moves(probs):
    #coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    coords = [i for i in range(len(probs))]
    coords.sort(key=lambda c: probs[c], reverse=True)
    coors = list(map(lambda x: utils.unflatten_coords(x), coords))
    return coors

def max_move_value(probs, moves=None):
    if len(probs)<1:
        return go.PASS, 0
    if not moves:
        coords = [i for i in range(len(probs))]
    else:
        coords = [utils.flatten_coords(m) for m in moves]
    im = max(coords, key=lambda x: probs[x])
    move = utils.unflatten_coords(im)
    #print("max:", move, im, probs[im])
    return move, probs[im]

def move_values(probs):
    coords = [i for i in range(len(probs))]
    coords.sort(key=lambda c: probs[c], reverse=True)
    coors = map(lambda x: (utils.unflatten_coords(x)), coords)
    for c,coor in zip(coords,coors):
        yield coor, probs[c]

def is_move_reasonable(board, move):
    # A move is reasonable if it is legal and doesn't fill in your own eyes.
    return board.is_move_legal(move)==0 

def select_random(board, forbid_pass=True):
    possible_moves = board.get_moves()
    if not possible_moves:
        return None
    move = random.choice(possible_moves)
    i = 0
    while forbid_pass and move == go.PASS:
        move = random.choice(possible_moves)
        i += 1
        if i>3:
            move = None
            break
    return move

def select_most_likely(board, move_probs, check_legal=True, forbid_pass=True):
    if forbid_pass:
        move_probs = move_probs[:-2]
    moves = sorted_moves(move_probs)
    rn = int(15*go.N*go.N/361)
    if check_legal:
        moves = [m for m in moves if (m==go.PASS or m==go.RESIGN or board.stones[m] == go.EMPTY)]
    move=None
    for m in moves:
        if board.step<rn:
            rd = random.choice(range(min(rn, len(moves))))
            move = moves[rd]
        else:
            move = m
        if board.is_move_legal(move)==0:
            return move
    return None

def select_weighted_random(board, moves):
    selection = random.randint(0, go.N**2+1)
    selected_move = moves[selection]
    if board.is_move_legal(move)==0:
        return selected_move
    else:
        # inexpensive fallback in case an illegal move is chosen.
        return select_most_likely(board, move_probabilities)

def simulate_game_random(board, forbid_pass=True):
    """Simulates a game to termination, using completely random moves"""
    pos = copy.deepcopy(board)
    while not pos.is_gameover:
        ill, caps = pos.play_move(select_random(pos, forbid_pass))
    return pos

def fast_score(board):
    pos=simulate_game(board, toend=True, forbid_pass=True, net=None)
    return pos.score()

def final_score(board, net):    
    pos = simulate_game(board, net = net)
    return pos.score()

def simulate_game(board, toend=True, forbid_pass=True, net=None):
    """Simulates a game starting from a board, using a policy network"""
    passbw = 0
    pos = copy.deepcopy(board)
    while config.running and (pos.step <= go.N*go.N*2):
        if net:
            move_probs, win_rate = net.run(pos)
            move = select_most_likely(pos, move_probs)
        else:
            move = select_random(pos, forbid_pass)
        if move is None or (not forbid_pass and move == go.PASS):
            passbw |= 1 if pos.to_move==go.BLACK else 2
            pos.play_move(go.PASS)
            if passbw==3:
                break
            else:
                continue

        ill, caps = pos.play_move(move)
        if ill>0:
            continue
        passbw = 0
    return pos

def simulate_many_games(policy1, policy2, boards):
    """Simulates many games in parallel, utilizing GPU parallelization to
    run the policy network for multiple games simultaneously.

    policy1 is black; policy2 is white."""

    # Assumes that all boards are on the same move number. May not be true
    # if, say, we are exploring multiple MCTS branches in parallel
    while boards[0].step <= POLICY_CUTOFF_DEPTH + POLICY_FINISH_MOVES:
        black_to_play = [pos for pos in boards if pos.to_move == go.BLACK]
        white_to_play = [pos for pos in boards if pos.to_move == go.WHITE]

        for policy, to_move in ((policy1, black_to_play),
                                (policy2, white_to_play)):
            all_move_probs = policy.run_many(to_move)
            for i, pos in enumerate(to_move):
                move = select_weighted_random(pos, all_move_probs[i])
                ill, caps = pos.play_move(move)

    for pos in boards:
        simulate_game_random(pos)

    return boards


class PlayerMixin:
    def suggest_move(self, board):
        return select_random(board)

class RandomPlayer:
    def __init__(self, **kwargs):
        self.running=True

    def suggest_move(self, board):
        self.root = Node(board)
        while config.running and self.running and not self.root.search_over:
            self.search_tree(self.root)
        move, move_values, maxc = pass_or_resign(self.root)
        if move==0:
            move = maxc.move
        return move, move_values, self.root.points

    def search_tree(self, node):
        if node.search_over:
            return
        if not node.childs:
            node.search_over = True
            score = fast_score(node.board)
            node.update(score if node.board.to_move==go.BLACK else -score)
            return
        selected = self.select(node)
        if not selected:
            node.search_over = True
            return
        for m,v in selected.moves.items():
            b = selected.board.copy()
            ill,caps = b.play_move(m)
            n=selected.add_child(b, selected, m, 0, None, 0)
            moves = b.get_moves()
            for m in moves:
                n.moves[m]=0
        self.search_tree(selected)

    def select(self, node):
        for child in node.childs:
            if not child.is_expanded or not child.search_over:
                return child
        return None

class AllSercher:
    def __init__(self, **kwargs):
        self.running=True
        self.root=None
        self.forbid_pass=True

    def suggest_move(self, board):
        if board.step<go.N:
            moves=board.get_moves()
            sel = random.choice(moves)
            mvs = [0 for s in go.ALL_COORDS]
            return sel, mvs, 0
        if board.step>go.N:
            self.root = self.get_node(board.recent[-1].move)
        if not self.root:
            mvs = [0 for s in go.ALL_COORDS]
            self.root = Node(board, None, go.PASS, 0, mvs, 0)
            self.expand(self.root)
        self.search_tree(self.root)
        move, move_values, maxc = pass_or_resign(self.root)
        if move==0:
            move = maxc.move
        return move, move_values, self.root.points

    def search_tree(self, node):
        if node.search_over:
            return
        if not node.childs:
            node.search_over = True
            score = fast_score(node.board)
            node.update(score if node.board.to_move==go.BLACK else -score)
            return
        for child in node.childs:
            if not child.is_expanded:
                self.expand(child)
                self.search_tree(child)
            elif not child.search_over:
                self.search_tree(child)

    def select(self, node):
        for child in node.childs:
            if not child.is_expanded or not child.search_over:
                return child
        return None

    def get_node(self, move):
        if not self.root:
            return None
        for child in self.root.childs:
            if child.move==move and (child.board.stones==self.board.stones).all() and not child.moves:
                r = child.copy()
                r.parent = None
                del self.root
                return r
        del self.root
        return None

    def expand(self, node):
        bs=[]
        mvs=[]
        vs=[]
        c = node.board.to_move
        for m,v in node.moves.items():
            if (m==go.PASS or m==go.RESIGN):
                if(self.forbid_pass):
                    continue
            b = node.board.copy()
            ill, caps=b.play_move(m)
            if ill>0:
                continue
            if b.is_gameover and m!=go.RESIGN:
                sc = fast_score(b)
                score = sc if b.to_move==go.BLACK else -sc
                node.add_child(b, m, v, None, score)
                continue
            bs.append(b)
            mvs.append(m)
            vs.append(v)
            #ms, points = self.net.run(b)  # 己胜率
        if not bs:
            node.moves.clear()
            return
        mss = [[0 for s in go.ALL_COORDS] for b in bs]
        points = [0 for b in bs]
        for b,m,v,ms,ps in zip(bs,mvs,get_noise(vs),mss,points):
            if self.forbid_pass:
                ms = ms[:-2]
            node.add_child(b, m, v, ms, ps)
        node.moves.clear()

class GreedyPolicyPlayerMixin:
    def __init__(self, net):
        self.net = net
        super().__init__()

    def suggest_move(self, board):
        move_probabilities, win_rate = self.net.run(board)
        return select_most_likely(board, move_probabilities)

class RandomPolicyPlayerMixin:
    def __init__(self, net):
        self.net = net
        super().__init__()

    def suggest_move(self, board):
        move_probabilities, win_rate = self.net.run(board)
        return select_weighted_random(board, move_probabilities)

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant

def UCT(board, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(board)

    for i in range(itermax):
        node = rootnode
        state = board.Copy()

        # Select
        node = node.select_child()
        if not node:
            break
        # Expand
        if node.moves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.moves)
            ill, caps=state.play_move(m)
            if ill>0:
                node.moves.remove(m)
                continue
            node = node.add_child(state,m,0,moves,0) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    else: print (rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited


c_PUCT = 3.8

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, board, parent=None, move=None, v=0.0, probs=None, points=0):
        self.board=board
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parent = parent  # "None" for the root node
        self.childs = []
        self.pv = v
        self.V=-go.N*go.N*2
        self.points = points
        self.visits = 0
        self.U = 0.0
        self.search_over = False
        self.moves={}
        if probs is not None and len(probs)>0:
            for m,v in move_values(probs):
                if m==go.PASS or m==go.RESIGN or board.stones[m] == go.EMPTY:
                    self.moves[m] = v  # future child nodes (m,v)

    @property
    def select_score(self):
       if self.search_over:
           return -999999.0
       return self.V+self.U

    def select_child(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        if not self.childs or self.search_over:
            return None
        node = self
        while config.running and node.is_expanded:
            if not node.childs:
                node.search_over = True
                return None
            if node.search_over:
                if node.parent:
                    node.parent.search_over = True
                return None
            node = max(node.childs, key=lambda c: c.select_score)
        return node

    def add_child(self, b, m, v, probs, points):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(b, parent=self, move=m, v=v,  probs=probs, points=points)
        del self.moves[m]
        self.childs.append(n)
        return n

    def update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.points += (result-self.points)/self.visits
        if self.parent:
            if self.V< go.N*go.N:
                self.V = result-self.parent.points
            else:
                self.V += (result-self.points) / self.visits/self.visits
            self.U = c_PUCT * math.sqrt(1 + self.parent.visits) * self.pv / (1 + self.visits)
            #print(self.V, self.U)
            self.parent.update(-result)

    def copy(self):
        r = Node(self.board, None, self.move, self.V, None, self.points)
        r.pv=self.pv
        r.childs=self.childs[:]
        r.visits=self.visits
        r.U=self.U
        return r

    @property
    def is_expanded(self):
        return True if self.childs or not self.moves else False

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.points) + "/" + str(self.visits) + " U:" + str(
            self.moves) + "]"

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

class MCTSSercher:
    def __init__(self, net, qp=None, outmsg=False):
        self.net = net
        self.qp=qp
        self.outmsg = outmsg
        self.root = None
        #self.score = 0
        self.would_pass = 0
        step = net.get_step()
        #print('train step={}'.format(step))
        self.vtime = config.mtcs_time + step // 100000
        self.can_pass = (random.randint(0,step)>3000)
        #self.ills = np.zeros((go.N,go.N), dtype=np.uint8)
        self.board = None
        self.width = 10
        self.depth = 1
        self.po=0
        self.max_po = 0
        if config.playouts>0:
            self.width = min(20, max(5, int(config.playouts*0.01)))
            self.depth = min(200, max(1, int(config.playouts*0.05)))

    def suggest_move(self, board, forbid_pass=True):
        self.po=0
        self.board=board
        self.forbid_pass=forbid_pass
        if board.step>0:
            self.root = self.get_node(board.recent[-1].move)
        if not self.root:
            move_probs, points = self.net.run(board)
            self.root = Node(board, None, go.PASS, 0, move_probs, points)
        else:
            points = self.root.points
            #print('使用上次的node')
        if config.playouts>0 and get_node_num(self.root)<MAX_NODES:
            self.max_po = config.playouts
        elif config.playouts>0:
            self.max_po = int(config.playouts/2)
        start = time.time()
        self.select_child(start)
        #if random.randint(0,1000)>998:
        #    print('第%d手，搜索了%d次' % (board.step, self.po))
        #self.printmsg("Searched for %s seconds" % (time.time() - start))
        #sorted_moves = sorted(self.root.children.keys(), key=lambda move, rt=self.root: rt.children[move].N, reverse=True)
        m,v,p,n = self.pass_or_resign()

        if m==0:
            maxv = go.N*go.N
            maxn = self.max_po*2.0 if self.max_po>0 else self.vtime*1000.0
            t = (math.tanh((go.N-board.step)/math.sqrt(go.N))+1) * 10.0 * maxv/maxn
            maxchild = max(self.root.childs, key=lambda x: x.V + t*x.visits)
            m = maxchild.move
            v = self.get_values()
            p = maxchild.points
            n = maxchild.visits
        #if random.randint(0,1000)>998:
        print("第{}手耗时{:.1f}秒搜索{}次，{}选择{}(V{:.1f}|N{})，局势：{}".format(board.step+1, 
                time.time()-start, self.po, go.get_color_str(board.to_move), 
                go.get_cmd_from_coor(m), v[utils.flatten_coords(m)], n,
                go.get_points_str(p, board.to_move)), file=sys.stderr)
        return m, v, p

    def select_child(self, t):
        if not self.root.is_expanded:
            self.expand(self.root)
        if not self.root.childs:
            return
        childs = sorted(self.root.childs, key=lambda x: x.select_score, reverse=True)
        child = None
        width = self.width
        if random.randint(0,100)>95:
            width -= 1
            child = random.choice(childs)
        topn = utils.take_n(width, childs)
        if child and child not in topn:
            topn.append(child)
        ts = time.time()
        for child in topn:
            if self.would_stop(t):
                break
            self.tree_search(child, t, 0, 1)
        while not self.would_stop(t, self.root):
            if self.root.search_over:
                print("root search_over")
                break
            self.tree_search(self.root, t, 0, self.depth)
            if (time.time()-ts)>3:
                ts = time.time()
                self.show_search()
                if self.qp and self.qp.show_heatmap:
                    self.qp.update_heatmap(self.get_values())

    def tree_search(self, root, start, depth=0, vdepth=1):
        t = time.time()
        selected = root.select_child()
        if self.po==5:
            print("while select time:", time.time()-t)
        if not selected:
            root.search_over = True
            return
        if not selected.is_expanded:
            t = time.time()
            self.expand(selected)
            depth += 1
            self.po += 1
            if self.po==5:
                print("expand time:", time.time()-t)
            t = time.time()
            selected.update(selected.points)
            if self.po==5:
                print("update time:", time.time()-t)
            if depth<vdepth and not self.would_stop(start, selected):
                self.tree_search(selected, start, depth, vdepth)
            return
        else:
            t = time.time()
            self.tree_search(selected, start, depth, vdepth)
            print("select time:", time.time()-t)

    def expand(self, node):
        bs=[]
        mvs=[]
        vs=[]
        c = node.board.to_move
        for m,v in node.moves.items():
            if (m==go.PASS or m==go.RESIGN):
                if(self.forbid_pass):
                    continue
            b = node.board.copy()
            ill, caps=b.play_move(m)
            if ill>0:
                continue
            if b.is_gameover and m!=go.RESIGN:
                sc = fast_score(b)
                score = sc if b.to_move==go.BLACK else -sc
                node.add_child(b, m, v, None, score)
                continue
            bs.append(b)
            mvs.append(m)
            vs.append(v)
            #ms, points = self.net.run(b)  # 己胜率
        if not bs:
            node.moves.clear()
            return
        mss, points = self.net.run_many(bs)
        for b,m,v,ms,ps in zip(bs,mvs,get_noise(vs),mss,points):
            if self.forbid_pass:
                ms = ms[:-2]
            node.add_child(b, m, v, ms, ps)
        node.moves.clear()

    def would_stop(self, t, node=None):
        if not config.running or (node and not node.childs):
            return True
        elif (self.root.visits>0):
            if (time.time()-t>self.vtime):
                return True
            if (self.max_po>0 and self.po>self.max_po):
                return True
        elif self.root.search_over:
            return True
        return False
    
    @property
    def is_same_color(self, board):
        return self.board.to_move == board.to_move

    def show_search(self):
        child = max(self.root.childs, key=lambda x: x.V)
        #for i,child in enumerate(childs):
        #    if i>10:
        #        break
        print("{}-{}({:.1f}|{})->".format(self.po, go.get_cmd_from_coor(child.move), 
                                        child.V, child.visits), end="")
        show_node_tree(child, 0)
        print(" ")

    def get_node(self, move):
        if not self.root:
            return None
        for child in self.root.childs:
            if child.move==move and (child.board.stones==self.board.stones).all() and not child.moves:
                r = child.copy()
                r.parent = None
                del self.root
                return r
        del self.root
        return None

    def get_values(self):
        vs = [0]*(go.N*go.N+2)
        for child in self.root.childs:
            m = utils.flatten_coords(child.move)
            vs[m] = child.V
        return vs

    def printmsg(self, msg, out=False):
        if self.outmsg or out:
            print(msg)

    def pass_or_resign(self):
        vs = self.get_values()
        if not self.root.childs and not self.root.moves:
            return go.PASS, vs, self.root.points, self.root.visits
        if (self.board.step<go.N*go.N/3 or self.forbid_pass):
            return 0, vs, self.root.points, self.root.visits
        point = 0.0
        p = True
        sumscore = 0.0
        for node in self.root.childs:
            if node.visits>0:
                sumscore += node.V
                if point!=0.0:
                    point = node.points
                if node.points!=point:
                    p=False
                    break
        if p:
            return go.PASS, vs, self.root.points, self.root.visits
        if sumscore+node.points<10:
            return go.RESIGN, vs, self.root.points, self.root.visits
        return 0, vs, self.root.points, self.root.visits

