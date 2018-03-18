import time
import go
import board
import strategies
import policy
import dataset
import utils
import os
import math
import random
import sgfparser
import copy
import threading
import shutil
import config
import numpy as np
import re
import sys
from scipy import stats
import multiprocessing as mp
from multiprocessing import Pool
import processplay
import allsearch

INIT_EPSILON = 1.0
FINAL_EPSILON = 0.0001
REINFORCE_SIZE = 160

def selftrain(qp=None, sgfdir=None, net=None):
    thtrain = TrainThread('增强学习训练', qp, sgfdir=sgfdir, net=net)
    thtrain.setDaemon(True)
    thtrain.start()
    return thtrain

def selftest(save_file='models', qp=None):
    thtrain = TrainThread('增强学习测试', qp)
    thtrain.setDaemon(True)
    thtrain.start()

class TrainThread (threading.Thread):
    def __init__(self, name, qp, sgfdir=None, net=None):
        threading.Thread.__init__(self)
        self.name=name
        self.qp=qp
        self.sgfdir=sgfdir
        self.train_net=net

    def run(self):
        print ("开启线程： " + self.name, file=sys.stderr)
        #sp = SelfPlay(qp=self.qp)
        config.running = True
        if '测试' in self.name:
            net = policy.PolicyNetwork(is_train=True, use_old=False)
            old_net = policy.PolicyNetwork(is_train=False, use_old=True)
            for i in range(config.test_num):
                test_play(net, old_net, qp=self.qp)
            net.save_variables()
            oldfile = old_net.save_file
            del old_net
            net.save(oldfile)
        else:
            #b=board.Board()
            #search = allsearch.AllSearch(b)
            #search.setDaemon(True)
            #search.start()
            self.self_train = SelfTrain(qp=self.qp, sgfdir=self.sgfdir, net=self.train_net)
            self.self_train.train_db(False)
        config.running=False
        print ("退出线程： " + self.name, file=sys.stderr)

    def reduce_lr(self):
        self.self_train.reduce_lr()

def test_play(net, old_net, qp=None, sgfdir=None, bw=go.BLACK):
    test_start = time.time()
    pos=board.Board(komi=7.5)
    if qp:
        qp.start(pos)
    move = None
    passbw = 0
    while config.running:
        #time.sleep(0.5)
        c = pos.to_move
        if pos.step>go.N*go.N*2:
            pos.result = 0
            msg = '双方超过722手，对局作为和棋结束。'
            if qp:
                qp.show_message(msg=msg)
            break
        if c==bw:
            move_probs, win_rate = net.run(pos)
        else:
            move_probs, win_rate = old_net.run(pos)
        pos.win_rate= win_rate*go.N*go.N
            
        move = strategies.select_most_likely(pos, move_probs, True, True)      #True:检查是否合法, True:拒绝PASS
        if move==go.RESIGN:
            pos.result = -go.N*go.N-1 if c==1 else go.N*go.N+1
            msg = '{}方第{}手认输，对局结束。{}中盘胜。'.format(go.get_color_str(c), pos.step, go.get_color_str(-c))
            if qp:
                qp.show_message(msg=msg)
            break
        if move is None or move == go.PASS:
            move = go.PASS
            passbw |= (1 if c==go.BLACK else 2)
            if passbw==3:
                pos.play_move(go.PASS)
                msg = '%s方第%d手PASS，对局结束。' % (go.get_color_str(c), pos.step)
                if qp:
                    qp.show_message(msg=msg)
                break
            else:
                pos.play_move(go.PASS)
                continue
        illegal, caps = pos.play_move(move)
        if illegal == 0:
            passbw = 0
            score = pos.score()
            if qp:
                qp.update(pos)
    if pos.step>go.N*go.N*2:
        return 0
    if abs(pos.result)<go.N*go.N:
        n = pos.step
        score = strategies.fast_score(pos)
        print("fast score: {}, n: {}/{}".format(score, n, pos.step), file=sys.stderr)
    else:
        score = pos.result*2
    score *= 1 if bw==go.BLACK else -1
    eloa = net.elo
    elob = old_net.elo
    ea=1/(1+10.0**((elob-eloa)/400))
    eb=1/(1+10.0**((eloa-elob)/400))
    win = 1 if score>0 else 0
    xs = 14.5 + 17500.0/(6000+net.elo)
    addea=xs*(win-ea)
    addeb=xs*(1-win-eb)
    print('old elo1: {:.2f}, 2: {:.2f}, add elo 1: {:.2f}, 2: {:.2f}'.format(
        eloa, elob, addea, addeb), file=sys.stderr)
    net.elo += addea
    net.elo = max(-5000, net.elo)
    old_net.elo += addeb
    old_net.elo = max(-5000, old_net.elo)
    pos.result = score / 2
    msg = '新权重：{},用时：{},{}手，{}方胜{:.1f}子, 增加{:.2f}/{:.2f}分。'.format(go.get_color_str(bw), 
        time.strftime("%Mm%Ss", time.localtime(float(time.time() - test_start))),
        pos.step+1, '新权重' if score>0 else '老权重', abs(score/2.0), addea, addeb)
    if qp:
        qp.show_message(msg=msg)
    if sgfdir:
        dt = sgfdir +'/test_' + time.strftime('%Y-%m-%d_%H_%M_%S')+'.sgf'
        msg = '%.1f：\t保存sgf棋谱文件到%s' % (time.time()-policy.start_time, dt)
        if qp:
            qp.show_message(msg=msg)
        if not os.path.exists(sgfdir):
            os.makedirs(sgfdir)
        sgfparser.save_board(pos, dt)
    return score/2

class SelfTrain():
    def __init__(self, qp=None, sgfdir='sgf', net=None):
        if net:
            self.net=net
        else:
            self.net = None
        self.board = board.Board() #[go.Position(to_move=go.WHITE) for i in range(1)]
        self.board.player1_name='ZiGo_New'
        self.board.player2_name='ZiGo_Old'
        self.qp=qp
        if self.qp:
            self.qp.start(self.board)
        self.sgfdir = sgfdir
        self.datas = dataset.DataSet()
        # neural net 1 always plays "black", and variety is accomplished by
        # letting white play first half the time.
        self.running=False

    def reduce_lr(self):
        self.net.change_leraning_rate(self.net.learning_rate*0.1)

    def train_sgf(self):
        train_start = time.time()
        self.datas.isloading = False
        config.running=True
        if not os.path.exists(self.datas.save_dir):
            os.makedirs(self.datas.save_dir)
        thtest = threading.Thread(target = self.thread_test)
        thtest.setDaemon(True)
        thtest.start()
        while config.running:
            nj=0
            for fn in os.listdir(self.sgfdir):
                filepath = os.path.join(self.sgfdir, fn)
                print('从%s中生成训练数据……' % (filepath), file=sys.stderr)
                if fn.endswith('.sgf'):
                    n = int(fn[2:-4])
                    if n!=self.net.train_n and n!= self.net.train_n+1:
                        continue
                    nj+=1
                    savepath ="{}/selftrain{}".format(self.datas.save_dir, n)
                    if not os.path.isdir(savepath):
                        os.makedirs(savepath)
                    else:
                        for fn in os.listdir(savepath):
                            fp = os.path.join(savepath, fn)
                            if os.path.isfile(fp) :
                                self.datas.data_files.append(fp)
                        continue
                    with open(filepath, 'r') as chunk_file:
                        file_content = chunk_file.read()
                        ms = re.findall(r'(\([\s\S]+?\))', file_content)
                    sp = self.datas.save_dir
                    self.datas.save_dir = savepath
                    for c in ms:
                        self.datas.add_from_file_content(c)
                        if not config.running:
                            break
                        if self.datas.data_size>=12800:
                            self.datas.shuffle(1)
                            self.datas.save()
                            self.datas.clear()
                    self.datas.save_dir = sp
            #self.thread_data()
            print('%s(%.1f)：%d个文件，开始训练……' % (time.strftime('%m-%d %H:%M:%S'),
                                                 time.time()-train_start, len(self.datas.data_files)))
            self.datas.start_load(del_file = False)
            self.net.game_num += nj*5000
            self.net.train_n += nj
            self.net.train(self.datas)
            self.datas.clear()
            self.datas.data_files=[]
        print('%s(%.1f)：已退出训练。' % (time.strftime('%m-%d %H:%M:%S'), time.time()-train_start))

    def train_db(self, test=False):
        train_start = time.time()
        config.running = True
        self.running = True
        if not self.net:
            self.net = policy.PolicyNetwork(is_train=True, selftrain=True)
        if test:
            thtest = threading.Thread(target = self.thread_test)
            thtest.setDaemon(True)
            thtest.start()
        self.datas.loaddb()
        self.net.train(self.datas)
        self.datas.clear()
        self.datas.data_files = []
        self.running = False


    def selftrain(self, test=True):
        train_start = time.time()
        #print('%s(%.1f)：正式开始训练……' % (time.strftime('%m-%d %H:%M:%S'), train_start-train_start))
        config.running = True
        self.running = True
        thsearch = threading.Thread(target=self.thread_allsearch)
        thsearch.setDaemon(True)
        thsearch.start()
        n=0
        #while config.running and self.running and n<100:
        #    time.sleep(5)
        if not self.net:
            self.net = policy.PolicyNetwork(is_train=True, selftrain=True)
        n = self.thread_data()
        if test:
            thtest = threading.Thread(target = self.thread_test)
            thtest.setDaemon(True)
            thtest.start()
        self.net.train(self.datas)
        self.datas.clear()
        self.datas.data_files = []
        self.net.train_n = n
        self.running = False

    def train_loop(self):
        test=False
        while config.running:
            #print(time.strftime('%m-%d %H:%M:%S'), "：开始自战……")
            if not self.net:
                self.net = policy.PolicyNetwork(is_train=True, selftrain=True)
                self.net.game_num += 2560
            #processplay.process_play(1, self.net, self.qp)
            self.thread_play()
            print(time.strftime('%m-%d %H:%M:%S'), "：开始训练……")
            self.train(test)
            if not test:
                test=True
        print("自我训练完成。")

    def play(self, pos=None, forbid_pass=True, sgfdir='sgf'):
        if not pos:
            pos = board.Board(komi=7.5)
        if self.qp:
            self.qp.start(pos)
        passbw = 0
        mcts = strategies.MCTSSercher(self.net, qp=self.qp)
        caps = None
        while config.running:
            if pos.step>go.N*go.N*2:
                pos.result=0
                break
            c = pos.to_move
            move, values, win_rate = mcts.suggest_move(pos, forbid_pass=forbid_pass)
            self.datas.add_from_node(pos, values, win_rate)
            if self.datas.data_size>12800:
                print("保存训练数据……", end="", file=sys.stderr)
                self.datas.save()
                self.datas.clear()
                print("完毕！", file=sys.stderr)
            pos.win_rate = win_rate
            if (move is None or move == go.PASS):
                passbw |= (1 if c == go.BLACK else 2)
                if passbw == 3 or (not forbid_pass):
                    pos.play_move(go.PASS)
                    score = strategies.fast_score(pos)
                    pos.result = score / 2
                    '''msg = '%s方第%d手PASS，对局结束, %s。' % (go.get_color_str(c),
                                                     pos.step, go.result_str(score / 2))

                    if self.qp:
                        self.qp.show_message(msg=msg)'''
                    break
                else:
                    pos.play_move(go.PASS)
                    continue
            elif move == go.RESIGN:
                pos.play_move(move)
                '''msg = '%s方第%d手投子认负，对局结束, %s。' % (go.get_color_str(c),
                                               pos.step, go.result_str(pos.result))

                if self.qp:
                    self.qp.show_message(msg=msg)'''
                break
            illegal, caps = pos.play_move(move)
            if illegal == 0:
                passbw = 0
                if self.qp:
                    self.qp.update(pos)

        if sgfdir:
            dt = sgfdir + '/self_' + time.strftime('%Y-%m-%d_%H_%M_%S') + '.sgf'
            '''msg = '%.1f：\t保存sgf棋谱文件到%s' % (time.time() - policy.start_time, dt)
            if self.qp:
                self.qp.show_message(msg=msg)'''
            if not os.path.exists(sgfdir):
                os.makedirs(sgfdir)
            sgfparser.save_board(pos, dt)
        return pos

    def thread_play(self):
        if not self.net:
            self.net = policy.PolicyNetwork(is_train=True, selftrain=True)
        game_num = self.net.game_num
        while config.running:
            train_start = time.time()
            forbid_pass=True if self.net.elo<-1000 else False
            if not forbid_pass and random.randint(100)>95:
                forbid_pass = True
            pos = self.play(forbid_pass=forbid_pass)
            #steps = sample_moves(board.step)
            #self.datas.add_from_pos_steps(board, steps, board.result)
            result = pos.result
            print('%s(%.1f)：第%d盘自我博弈完成，%s胜%.1f子。' % (time.strftime(
                '%m-%d %H:%M:%S'), time.time()-train_start, self.net.game_num-game_num+1, 
                go.get_color_str(result), abs(result)))
            self.net.game_num += 1
            if self.net.game_num-game_num>1000:
                break

    def thread_play_random(self, num=1000):
        game_num = 0
        data = dataset.DataSet()
        name = mp.current_process().name
        while config.running:
            train_start = time.time()
            pos = strategies.simulate_game(board.Board(komi=7.5))
            print("name:", name, "{}：第{}盘随机盘面已准备就绪，开始复盘……".format(time.time()-train_start, game_num))
            train_start = time.time()
            self.replay(pos, data)
            print("name:", name, "{}：第{}盘复盘完成。".format(time.time()-train_start, game_num))
            game_num += 1
            if game_num>num:
                break
        if self.data.data_size>256:
            print("name:", name, "保存训练数据……", end="")
            data.save()
            print("name:", name, "保存完毕！")

    def thread_test(self):
        print("开始自我测试……")
        if not self.net:
            self.net = policy.PolicyNetwork(is_train=True, selftrain=True)
        old_net = policy.PolicyNetwork(is_train=False, use_old=True)
        config.maxelo=old_net.elo + 30 + 20000 / (6000+self.net.elo)
        st=time.time()
        win=0
        i=0
        self.running=True
        while config.running and self.running:
            result = test_play(self.net, old_net, qp=self.qp, bw = go.BLACK if i%2==0 else go.WHITE)
            win += 1 if result>0 else 0
            i += 1
            wr = win*100.0/i
            print("第{}盘{}，胜率：{:.2f}，{:.1f}盘/分钟。".format(i,
                go.result_str(result).replace("黑","新权重").replace("白","老权重"), wr, 
                60/(time.time()-st)), file=sys.stderr)
            st=time.time()
            time.sleep(0.1)
            up_elo = 30 + 20000 / (6000+self.net.elo)
            if self.net.elo>config.maxelo+up_elo:
                self.net.save_variables()
                old_net.restore()
                self.net.save(old_net.save_file)
                config.maxelo=self.net.elo
                i = 0
                win = 0
            if i>400:
                i=0
                win=0
        self.running = False
        print("退出自我测试。")
        
    def thread_data(self):
        maxn = 0
        fns = os.listdir(self.datas.save_dir)
        if not fns:
            return maxn
        fn = max(fns, key=lambda x:int(x[5:]))
        maxn = int(fn[5:])
##        for fn in os.listdir(self.datas.save_dir):
##            filepath = os.path.join(self.datas.save_dir,fn)
##            if os.path.isfile(filepath):
##                m = re.match(r'train([0-9]+)', filepath)
##                if m:
##                    n = int(m.groups(1))
##                    if n>maxn:
##                        maxn=n
        if not self.net:
            self.net = policy.PolicyNetwork(is_train=True, selftrain=True)
        s =  self.net.train_n
        n = int(250000*config.batch_size/10000/16)
        if s + n>maxn:
            s=max(0,min(s, maxn-n))
            e=maxn
        else:
            e = min(maxn, s + n)
        print("最大文件号：", maxn, "训练开始文件号：", s, "结束：", e)
        self.datas.data_files = []
        for i in range(s, e):
            filepath = os.path.join(self.datas.save_dir, "train{}".format(i))
            if os.path.isfile(filepath):
                self.datas.data_files.append(filepath)
                self.net.game_num += 80
        fnum = len(self.datas.data_files)
        if fnum<=0:
            config.running=False
        print("训练文件夹：", self.datas.save_dir, "文件数：", fnum)
        self.datas.start_load(del_file=False)
        return e+1

    def thread_allsearch(self):
        b=board.Board()
        search = allsearch.AllSearch(b)
        #search.setDaemon(True)
        search.start()

    def replay(self, pos, data=None, times=80):
        final_score = strategies.fast_score(pos)
        mcts = strategies.RandomPlayer()
        rboard = pos.copy()
        name = mp.current_process().name
        for i in range(times):
            if not config.running:
                break
            rboard.undo()
            curboard = rboard.copy()
            c=curboard.to_move
            while config.running and not curboard.is_gameover:
                move, move_values, points = mcts.suggest_move(curboard)
                curboard.win_rate = points if c == go.BLACK else -points
                data.add_from_node(curboard, move_values, points)
                if data.data_size%1024==5:
                    print("name:", name, ", data size:", data.data_size, file=sys.stderr)
                if data.data_size>25600:
                    print("name:", name, "保存训练数据……", end="")
                    data.save()
                    data.clear()
                    print("name:", name, "保存完毕！")
                curboard.play_move(move)

            score = curboard.win_rate if c==go.BLACK else -curboard.win_rate
            cscore = score-final_score
            cscore = cscore  if c == go.BLACK else -cscore
            if i%20==15:
                print("name:", name, "倒退{}步，全探查结果：{}，{}{}了{:.1f}子".format(i+1, 
                    go.result_str(score), go.get_color_str(c), 
                    "增加" if cscore>0 else "减少", abs(cscore)), file=sys.stderr)

def sample_moves(max_step):
    n = random.randint(min(config.sample_num, max_step), max_step)
    l = list(range(max_step))
    random.shuffle(l)
    return l[0:n]

def extract_moves(final_board, n=strategies.POLICY_CUTOFF_DEPTH):
    wins = []
    loss = []
    win_color = go.BLACK if final_board.score()>0 else go.WHITE
    start = 0 if n>0 else final_board.step+n
    end = n if n>0 else final_board.step
    for i in range(start, end):
        r = final_board.recent[i]
        if r.color == win_color:
            wins.append(i)
        else:
            loss.append(i)
    return wins, loss

def replay_moves(final_board, better_board, c, n, step_num):
    bend = min(n+step_num, better_board.step)
    fend = min(n+step_num, final_board.step)
    if better_board.get_color(n) == c:
        winning_steps = list(range(n, bend, 2))
        losing_steps = list(range(n, fend, 2))
    else:
        winning_steps = list(range(n+1, bend, 2))
        losing_steps = list(range(n+1, fend, 2))
    return winning_steps, losing_steps


