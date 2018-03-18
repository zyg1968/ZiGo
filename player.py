import policy
import threading
import time
import sys
from strategies import *

policy_names=['policy', 'ramdom', 'mtcs']
PLAYER, POLICY, GTP = 0, 1, 2

class Player():
    def __init__(self, name, time=20, net=None, color=1, gtpon=False):
        self.name=name
        self.time=time
        name = name.lower()
        self.passed = 0
        self.color = color
        self.net = net
        if name in policy_names:
            self.player_type = POLICY
            self.net=policy.PolicyNetwork(is_train=False) if net is None else net
            if name == policy_names[2]:
                self.mcts = MCTSPlayer(self.net)
        else:
            self.player_type = PLAYER

    def get_move(self, board, caps=None):
        assert(board.to_move == self.color)
        name = self.name.lower()
        if name not in policy_names:
            return None
        start = time.time()
        all_move_probs, win_rate = self.net.run(board)
        if name == policy_names[0]:
            move = select_most_likely(board, all_move_probs)
        elif name == policy_names[1]:
            move = select_weighted_random(board, all_move_probs)
        elif name == policy_names[2]:
            move, all_move_probs, win_rate = self.mcts.suggest_move(board, caps)
        return move, win_rate
            #board.play_move(move, board.to_move)


class InputThread (threading.Thread):
    def __init__(self, name, playman):
        threading.Thread.__init__(self)
        self.name = name
        self.playman = playman

    def run(self):
        print ("开启线程： " + self.name)
        if self.name.startswith('命令行'):
            self.waitcmd()
        else:
            self.waitmessage()
        print ("退出线程： " + self.name)

    def waitcmd(self):
        while (not self.playman.engine.disconnect) and self.playman.running:
            c = self.playman.gomanager.to_move
            cmd = input()
            print('你输入了命令：%s' % (cmd))
            engine_reply = self.playman.engine.send(cmd)
            sys.stdout.write(engine_reply)
            if engine_reply.startswith('='):
                self.playman.engine.make_move(parse_move(cmd))
                strs=cmd.upper().split(' ')
                if len(strs)<2:
                    return
                color, vertex=gtp.parse_move(strs[1:])
                x,y=go.get_coor_from_vertex(vertex)
                print('你命令走子成功：%s(%d,%d)' % (strs[-1], x, y))
                self.playman.qp.update(self.playman.gomanager)
                self.playman.qp.havemoved = True

    def waitmessage(self):
        pass
