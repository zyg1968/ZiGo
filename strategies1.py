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

# Draw moves from policy net until this threshold, then play moves randomly.
POLICY_CUTOFF_DEPTH = int(go.N * go.N * 1.0) # 270 moves for a 19x19
# This speeds up the simulation, and it also provides a logical cutoff
# for which moves to include for reinforcement learning.
# However, some situations end up as "dead, but only with correct play".
POLICY_FINISH_MOVES = int(go.N * go.N * 0.2) # 72 moves for a 19x19
# Random play can destroy the subtlety of these situations, so we'll play out
# a bunch more moves from a smart network before playing out random moves.

def sorted_moves(probability_array):
    #coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    coords = [i for i in range(go.N*go.N)]
    coords.sort(key=lambda c: probability_array[c], reverse=True)
    coors = list(map(lambda x: utils.unflatten_coords(x), coords))
    return coors

def is_move_reasonable(board, move):
    # A move is reasonable if it is legal and doesn't fill in your own eyes.
    return board.is_move_legal(move)==0 

def select_random(board, forbid_pass=True):
    all_coords = [(i, j) for i in range(go.N) for j in range(go.N)]
    possible_moves = [m for m in all_coords if board.stones[m]==go.EMPTY]
    random.shuffle(possible_moves)
    for move in possible_moves:
        if (not forbid_pass or move!=go.PASS) and board.is_move_legal(move)==0:
            return move
    return go.PASS

def select_most_likely(board, move_probs, check_legal=True, forbid_pass=True):
    moves = sorted_moves(move_probs)
    if check_legal:
        moves = [m for m in moves if (m==go.PASS or board.stones[m] == go.EMPTY)]
    for i in range(len(moves)):
        rd = i
        if board.n<30:
            rd = random.randint(0,30)
        move = moves[rd]
        if (forbid_pass or board.n<100) and move == go.PASS:
            continue
        if not check_legal or board.is_move_legal(move) == 0:
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
    while not (pos.recent[-2].move == go.PASS and pos.recent[-1].move == go.PASS):
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
    while toend or pos.n <= go.N*go.N*0.8:
        if net:
            move_probs, win_rate = net.run(pos)
            move = select_most_likely(pos, move_probs)
        else:
            move = select_random(pos, forbid_pass)
        if move is None or move == go.PASS:
            passbw |= 1 if pos.to_play==go.BLACK else 2
            if passbw==3:
                break
            else:
                pos.pass_move()
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
    while boards[0].n <= POLICY_CUTOFF_DEPTH + POLICY_FINISH_MOVES:
        black_to_play = [pos for pos in boards if pos.to_play == go.BLACK]
        white_to_play = [pos for pos in boards if pos.to_play == go.WHITE]

        for policy, to_play in ((policy1, black_to_play),
                                (policy2, white_to_play)):
            all_move_probs = policy.run_many(to_play)
            for i, pos in enumerate(to_play):
                move = select_weighted_random(pos, all_move_probs[i])
                ill, caps = pos.play_move(move)

    for pos in boards:
        simulate_game_random(pos)

    return boards


class PlayerMixin:
    def suggest_move(self, board):
        return select_random(board)

class RandomPlayerMixin:
    def suggest_move(self, board):
        return select_random(board)

class GreedyPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, board):
        move_probabilities, win_rate = self.policy_network.run(board)
        return select_most_likely(board, move_probabilities)

class RandomPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, board):
        move_probabilities, win_rate = self.policy_network.run(board)
        return select_weighted_random(board, move_probabilities)

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
# Exploration constant
c_PUCT = 5

class MCTSNode():
    '''
    A MCTSNode has two states: plain, and expanded.
    An plain MCTSNode merely knows its Q + U values, so that a decision
    can be made about which MCTS node to expand during the selection phase.
    When expanded, a MCTSNode also knows the actual board at that node,
    as well as followup moves/probabilities via the policy network.
    Each of these followup moves is instantiated as a plain MCTSNode.
    '''
    @staticmethod
    def root_node(board, move_probabilities, ills):
        node = MCTSNode(None, None, 0)
        node.board = board
        node.expand(move_probabilities, ills)
        return node

    def __init__(self, parent, move, prior):
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior;
        self.board = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.Q = self.parent.Q if self.parent is not None else 0    #黑方平均胜率
        self.U = prior # monte carlo exploration bonus
        self.N = 0 # number of times node was visited
        self.W = 0  #黑棋胜率和
        self.expanded_childs = 0

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def action_score(self):
        # Note to self: after adding value network, must calculate 
        # self.Q = weighted_average(avg(values), avg(rollouts)),
        # as opposed to avg(map(weighted_average, values, rollouts))
        if self.board:
            c = self.board.to_play
        elif self.parent and self.parent.board:
            c = -self.parent.board.to_play
        else:
            return -1
        return self.Q*c + self.U

    def is_expanded(self):
        return self.board is not None

    def compute_board(self):
        self.board, ill = self.parent.board.try_move(self.move, None)
        self.prior = (1-0.25)*self.prior + 0.25* (random.randint(0,1) \
            if random.randint(0, self.board.n if self.board else 300)<50 else 1);
        return self.board, ill

    def expand(self, move_probabilities, ills):
        for i, prob in enumerate(move_probabilities):
            move = (divmod(i, go.N))
            il = 1 if self.board.to_play == go.BLACK else 2
            if move!=go.PASS and self.board.stones[move] == go.EMPTY and ills[move] & il == 0:
                self.children[move] = (MCTSNode(self, move, prob))
        #self.children.sort(key=lambda x: x.prior, reverse = True)

    def backup_value(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W/self.N  #self.Q + (value*self.board.to_play - self.Q) / self.N
        if self.parent:
            self.U = c_PUCT * math.sqrt(self.parent.N) * self.prior / (1+self.N)
            # must invert, because alternate layers have opposite desires
            self.parent.backup_value(value)

    def select_leaf(self):
        current = self
        while current.is_expanded():
            if len(current.children)<1:
                return None
            current = max(current.children.values(), key=lambda node: node.action_score)
        return current


class MCTSPlayer:
    def __init__(self, policy_network, outmsg=False):
        self.policy_network = policy_network
        self.outmsg = outmsg
        self.root = None
        self.t = 1
        #self.score = 0
        self.would_pass = 0
        step = policy_network.get_step()
        print('train step={}'.format(step))
        self.vtime = config.mtcs_time + step // 100000
        self.can_pass = (random.randint(0,step)>3000)
        #super().__init__()
        self.ills = np.zeros((go.N,go.N), dtype=np.uint8)

    def suggest_move(self, board, caps):
        if caps:
            for m in caps:
                self.ills[m] = 0
        if board.n>30:
            self.t=0.05
        if board.n>0:
            self.root = self.get_node(board)
        if not self.root:
            move_probs, win_rate = self.policy_network.run(board)
            self.root = MCTSNode.root_node(board, move_probs, self.ills)
        else:
            win_rate = self.root.Q
            print('使用上次的node')
        start = time.time()
        ts=self.select_child(start)
        if random.randint(0,1000)>997:
            print('第%d手，搜索了%d次' % (board.n, ts))
        #self.printmsg("Searched for %s seconds" % (time.time() - start))
        #sorted_moves = sorted(self.root.children.keys(), key=lambda move, rt=self.root: rt.children[move].N, reverse=True)
        self.would_pass = 0
        winnode = None
        pi = np.zeros(go.N*go.N+1, dtype=np.float32)
        if self.can_pass and self.root.children:
            if board.n>100 and board.recent[-1].move == go.PASS:
                self.would_pass += 2
            if (win_rate+1)/2< config.vresign or (win_rate+1)/2>(1-config.vresign):
                self.would_pass += 2
            winnode = max(self.root.children.values(), key=lambda x: x.Q)
            if (winnode.Q*board.to_play+1)/2 < config.vresign or (winnode.Q*board.to_play+1)/2>(1-config.vresign):
                self.would_pass += 2
            self.would_pass += board.n // 100
        if not self.root.children or (self.can_pass and self.would_pass>5):
            pi[go.N*go.N] = 1
            return go.PASS, pi, winnode.Q if winnode else 0
        alln = reduce(lambda x,y: x+y.N, self.root.children.values(), 0)
        if alln == 0:
            alln = 1
        for m, node in self.root.children.items():
            ind = utils.flatten_coords(m)
            pi[ind] = node.N**(1/self.t)/alln**(1/self.t)
        maxnode = max(self.root.children.values(), key=lambda x: x.N**(1/self.t)/alln**(1/self.t))
        #if self.root.Q<self.vresign and select_node.Q<self.vresign:
        #    return go.PASS, 0, 0
        return maxnode.move, pi, maxnode.Q      #黑棋胜率

    def select_child(self, t):
        j=0
        #childs = sorted(self.root.children.values(), key=lambda x: x.action_score, reverse=True)
        #if random.randint(0,100)>95:
        #    child = random.choice(childs)
        #    if not child.is_expanded():
        #        self.tree_search(self.root, child)
        #        j += 1
        #for v in childs:
        #    if time.time()-t>self.vtime and self.root.expanded_childs>0:
        #        break
        #    if not v.is_expanded():
        #        self.tree_search(self.root, v)
        #    j += 1
        #    if j>config.mtcs_width:
        #        break
        #if j<1:
        #    return j
        while time.time() - t < self.vtime or (len(self.root.children)>0 and self.root.expanded_childs<1):
            self.tree_search(self.root)
            j += 1
        return j

    def tree_search(self, root):
        # selection
        chosen_leaf = root.select_leaf()
        if not chosen_leaf:
            return
        #depth -= 1
        #if depth<1 or time.time()-t>self.vtime or not chosen_leaf:
        #    if wr is not None:
        #        win_rate = wr
        #    elif chosen_leaf:
        #        ms, win_rate = self.policy_network.run(chosen_leaf.board)      #己胜率
        #        win_rate *= chosen_leaf.board.to_play           #变成黑胜率
        #    else:
        #        ms, win_rate = self.policy_network.run(root.board)      #己胜率
        #        win_rate *= root.board.to_play
        #    root.backup_value(win_rate)
        #    return
        # expansion
        board, ill = chosen_leaf.compute_board()
        if board is None:
            c = chosen_leaf.parent.board.to_play
            self.ills[chosen_leaf.move] |= 1 if c == go.BLACK else 2
            del chosen_leaf.parent.children[chosen_leaf.move]
            return
        move_probs, win_rate = self.policy_network.run(board)
        win_rate *= board.to_play       
        chosen_leaf.expand(move_probs, self.ills)
        chosen_leaf.parent.expanded_childs += 1
        chosen_leaf.backup_value(win_rate)
        # evaluation
        #tree_search(chosen_leaf, depth, t, wr=win_rate)

    def estimate_value(self, chosen_leaf, random=True):
        pos = simulate_game(chosen_leaf.board, net = None if random else self.policy_network)
        return pos.score()

    def get_node(self, move):
        if not self.root:
            return None
        for k, v in self.root.children.items():
            if k==move and v.is_expanded():
                return v
        return None

    def printmsg(self, msg, out=False):
        if self.outmsg or out:
            print(msg)
