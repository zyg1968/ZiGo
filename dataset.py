import numpy as np
import math
import go
import utils
import selfplay
import gzip
import struct
import threading
import tensorflow as tf
import sgfparser
import os
import time
import config
import random
import sys
import re
import shufflenet
import dbhash

def softmax(x):
    return (np.exp(x)/np.sum(np.exp(x),axis=0)).tolist()

def sample_moves(max_step, min_step):
    n = random.randint(min(min_step, max_step), max_step)
    l = list(range(max_step))
    random.shuffle(l)
    return l[0:n]

def move_value_norm(v):
    return math.tanh(v*10.0/(go.N*go.N))

def make_onehot(coord):
    output = np.zeros([go.N ** 2+2], dtype=np.float32)
    output[utils.flatten_coords(coord)] = go.N * go.N
    return output

def get_feature_from_stones(stoneses, to_move):
    history=int((config.FEATURE_NUM-config.PLAY_FEATURES)/config.STATE_FEATURES)
    onehot_features = np.zeros([config.FEATURE_NUM, go.N, go.N], dtype=np.int8)
    for i,stones in enumerate(stoneses):
        if i>=history:
            break
        if config.STATE_FEATURES == 1:
            onehot_features[i] = stones
        else:
            onehot_features[i*2,stones==c] = 1
            onehot_features[i*2+1,stones==-c] = 1
    if config.PLAY_FEATURES == 2:
        if to_move==go.BLACK:
            onehot_features[config.FEATURE_NUM-2, :,:] = 1
        else:
            onehot_features[config.FEATURE_NUM-1,:,:] = 1
    elif config.PLAY_FEATURES == 1:
        onehot_features[config.FEATURE_NUM-1, :,:] = to_move
    return onehot_features

def get_feature(board):
    return get_features(board, board.step)

def get_features(pos, step):
    history=int((config.FEATURE_NUM-config.PLAY_FEATURES)/config.STATE_FEATURES)
    onehot_features = np.zeros([config.FEATURE_NUM, go.N, go.N], dtype=np.int8)
    for i in range(history):
        c = pos.get_color(step-i)
        if len(pos.recent)<1 or pos.step == step-i:
            b = pos.stones
        elif step-i<1:
            b=pos.recent[0].stones
        elif step - i >= pos.step-1:
            b=pos.recent[-1].stones
        else:
            b = pos.get_stones(step-i)
        if config.STATE_FEATURES == 1:
            onehot_features[i] = b
        else:
            onehot_features[i*2,b==c] = 1
            onehot_features[i*2+1,b==-c] = 1
    c = pos.get_color(step)
    if config.PLAY_FEATURES == 2:
        if c==go.BLACK:
            onehot_features[config.FEATURE_NUM-2, :,:] = 1
        else:
            onehot_features[config.FEATURE_NUM-1,:,:] = 1
    elif config.PLAY_FEATURES == 1:
        onehot_features[config.FEATURE_NUM-1, :,:] = c
    return onehot_features

class DataSet():
    def __init__(self, states=None, move_values=None, win_rates=None):
        #self.states = states if states else np.ndarray([0,config.FEATURE_NUM,go.N,go.N], dtype=np.int8)
        #self.move_values = move_values if move_values else np.ndarray([0,go.N*go.N+1], dtype=np.float32)
        #self.win_rates = win_rates if win_rates else np.ndarray([0], dtype=np.float32)
        self.states = states if states else []
        self.move_values = move_values if move_values else []
        self.win_rates = win_rates if win_rates else []
        self.data_size = len(self.states)
        if self.data_size>0:
            self.board_size = len(self.states[0])
        else:
            self.board_size = go.N
        self.nodata=False
        self.save_dir = '%s/f%d' % (config.data_dir, config.FEATURE_NUM)
        self.num_saved = self.get_save_num()
        self.index = 0
        self.isloading = False
        self.batch_size = config.batch_size
        self.data_files = []
        self.event = threading.Event()
        self.event.set()
        self.loadth=None

    def clear(self):
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        #self.states = np.ndarray([0,config.FEATURE_NUM,go.N,go.N], dtype=np.int)
        #self.move_values = np.ndarray([0,go.N*go.N+1], dtype=np.float32)
        #self.win_rates = np.ndarray([0], dtype=np.float32)
        self.states = []
        self.move_values = []
        self.win_rates = []
        self.data_size = 0
        self.event.set()

    def get_save_num(self):
        for fn in os.listdir(self.save_dir):
            m = re.match(r"train([0-9]+)", fn)
            if m:
                num = m.group(1)
                num = int(num)
                return num
        return 0

    def del_files(self):
        for fn in self.data_files:
            os.remove(fn)
        self.data_files = []
        self.index = 0
        self.num_saved = 0

    def shuffle(self, pct=1):
        k = int(self.data_size/self.batch_size*pct)*self.batch_size
        if k<self.batch_size:
            return
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        data = list(zip(self.states, self.move_values, self.win_rates))
        data1 = random.sample(data, k)
        self.states[:],self.move_values[:],self.win_rates[:]=zip(*data1)
        self.data_size=len(self.win_rates)
        self.event.set()

    def get_batch(self, batch_size):
        if self.data_size<batch_size:
            return None,None,None
        s = 0
        e = batch_size
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        mvs = self.move_values[s:e]
        boards2 = go.MAX_BOARD*go.MAX_BOARD
        add_p = config.policy_size-boards2
        for mv in mvs:
            if len(mv)==go.N*go.N:
                mv.append(0.0)
                mv.append(0.0)
            for i in range(go.N*go.N, boards2):
                mv.insert(-add_p, 0.0)
        mvs = np.array(mvs)
        ps = np.array(self.win_rates[s:e])
        r = (np.array(self.states[s:e]), mvs,
             ps.reshape(batch_size, 1))
        self.states = self.states[e:]
        self.move_values = self.move_values[e:]
        self.win_rates = self.win_rates[e:]
        self.data_size = len(self.win_rates)
        self.event.set()
        return r

    def add_step(self, board, step):
        pm = board.recent[step]
        move = pm.move
        if not move:
            move=go.PASS
        if step<200 and move==go.PASS:
            return
        color = pm.color
        states = get_features(board, step)
        move_values = pm.values
        if pm.values is None:
            move_values = make_onehot(move)
        win = board.scores
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        self.states.append(states)
        self.move_values.append(move_values)    #.reshape(-1, go.N*go.N+1)
        #self.win_rates = np.append(self.win_rates, win)
        self.win_rates.append(win)
        self.data_size = len(self.win_rates)
        self.event.set()

    def add_from_dataset(self, datas):
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        if self.states is not None:
            #self.states = np.concatenate((self.states, datas.states), axis=0)
            #self.move_values = np.concatenate((self.move_values, datas.move_values), axis=0)
            #self.win_rates = np.concatenate((self.win_rates, datas.win_rates), axis=0)
            self.states += datas.states
            self.move_values += datas.move_values
            self.win_rates += datas.win_rates
            self.data_size = len(self.states)
        else:
            self = datas
        self.event.set()

    def add_from_node(self, board, move_values, points):
        states = get_feature(board)
        #for i in range(go.N*go.N, MAX_BOARD*MAX_BOARD-go.N*go.N):
        #    move_values.insert(-2,0)
        win_rates = points/(go.N*go.N)
        while not self.event.is_set():
            self.event.wait(0.1)
        self.event.clear()
        self.states.append(states)
        self.move_values.append(softmax(np.array(move_values)))
        self.win_rates.append(win_rates)
        self.data_size = len(self.win_rates)
        self.event.set()

    def add_from_pos_steps(self, board, steps, final_score, train=False, act=None):
        assert board.step == len(board.recent), "Position history is incomplete"
        steps.sort()
        for step in steps:
            if step<10 or step>board.step-1:
                continue
            self.add_step(board, step)

    def add_from_file(self, fn):
        sgf = sgfparser.SgfParser(filename=fn)
        #pos = sgfparser.get_sgf_board(fn)
        if sgf.content:
            for pos in sgf.replay_sgf():
                step = pos.step-1
                if step<10:
                    continue
                self.add_step(pos, step)
                
    def add_from_file_content(self, content):
        sgf = sgfparser.SgfParser(content=content)
        #pos = sgfparser.get_sgf_board(fn)
        if sgf.content:
            for pos in sgf.replay_sgf():
                step = pos.step-1
                if step<10:
                    continue
                self.add_step(pos, step)

    def add_from_board(self, pos, n=0):
        if n==0:
            steps = list(range(pos.step))
        else:
            steps = sample_moves(pos.step, n)
        self.add_from_pos_steps(pos, steps, pos.result*2)

    def start_load(self, del_file=False):
        if self.loadth and self.loadth.is_alive():
            return
        self.isloading=True
        self.event.set()
        random.shuffle(self.data_files)
        self.loadth = threading.Thread(target = self.load, args=(del_file,))
        self.loadth.setDaemon(True)
        self.loadth.start()

    def save(self, fn=None):
        if self.data_size<1:
            return
        if not fn:
            fn = self.save_dir+'/train'+str(self.num_saved)
            while os.path.isfile(fn):
                if fn not in self.data_files:
                    self.data_files.append(fn)
                self.num_saved += 1
                fn = self.save_dir+'/train'+str(self.num_saved)

        path = os.path.dirname(fn)
        if not os.path.exists(path):
            os.makedirs(path)
        print('保存训练数据到%s中……' % (fn), file=sys.stderr)
        batchs = math.ceil(self.data_size/self.batch_size)
        header_bytes = struct.pack('ii', self.data_size, self.batch_size)
        with gzip.open(fn, "wb", compresslevel=6) as f:
            f.write(header_bytes)
            for i in range(batchs):
                s=i*self.batch_size
                e=min((i+1)*self.batch_size, self.data_size)
                np.save(f, np.array(self.states[s:e]))
                np.save(f, np.array(self.move_values[s:e]))
                np.save(f, np.array(self.win_rates[s:e]))
        self.num_saved += 1
        self.data_files.append(fn)
        print('训练数据已保存到%s中，目前文件%d个' % (fn, len(self.data_files)), file=sys.stderr)

    def load(self, del_file=True):
        self.isloading = True
        print('读训练数据文件线程启动，共{}文件，持续喂数据中……'.format(len(self.data_files)), file=sys.stderr)
        for fn in self.data_files:
            if not config.running or not self.isloading:
                break
            if not os.path.isfile(fn):
                continue
            while self.data_size>=config.batch_size*32:
                if not config.running or not self.isloading:
                    return
                time.sleep(1)
            print('读取文件%s中的数据……' % (fn), file=sys.stderr)
            with gzip.open(fn, "rb") as f:
                header_bytes = f.read(struct.calcsize('ii'))
                data_size, batch_size = struct.unpack('ii', header_bytes)
                batch_num = math.ceil(data_size/batch_size)
                for i in range(batch_num):
                    if not config.running or not self.isloading:
                        return
                    states = np.load(f) 
                    move_values = np.load(f) 
                    win_rates = np.load(f)
                    while not self.event.is_set():
                        self.event.wait(0.1)
                    self.event.clear()
                    self.states += states.tolist()
                    self.move_values += move_values.tolist()
                    self.win_rates += win_rates.tolist()
                    self.data_size = len(self.states)
                    self.event.set()
                assert len(f.read()) == 0
                f.close()
            if del_file:
                os.remove(fn)
        self.isloading = False
        self.data_files = []
        print('数据已全部读入，等待训练完成……', file=sys.stderr)

    def loaddb(self):
        self.isloading = True
        sqldb = dbhash.SqliteDb()
        cnt = sqldb.get_count()
        print('读训练数据库，共{}条，持续喂数据中……'.format(cnt), file=sys.stderr)
        for row in sqldb.load_data():
            if not config.running or not self.isloading:
                break
            if not row:
                continue
            while not self.event.is_set():
                self.event.wait(0.1)
            self.event.clear()
            self.states += row[0]
            self.move_values += row[1]
            self.win_rates += row[2]
            self.data_size = len(self.states)
            self.event.set()
        self.isloading = False
        self.data_files = []
        print('数据已全部读入，等待训练完成……', file=sys.stderr)

