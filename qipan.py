import threading
import time
import numpy as np
from tkinter import *
from scrolltext import ScrollText
from PIL import Image, ImageTk
import go
import board
import sys
import tkinter.filedialog
import sgfparser
import main
import config
import os
import subprocess
import intergtp
import dataset
import dbhash

def get_process_count(proname):
    p = os.popen('tasklist /FI "IMAGENAME eq %s"' % (proname))
    return p.read().count(proname)


class QiPan(object):
    def __init__(self, name, root, bj, bs, ws, hqh, bqh, last):
        # threading.Thread.__init__(self)
        self.name = name
        # self.queue=msgqueue
        self.root = root
        self.img1 = bj
        self.img2 = bs
        self.img3 = ws
        self.img4 = hqh
        self.img5 = bqh
        self.img6 = last
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.height = self.screen_height
        self.board = None
        self.clicked = None
        self.message = None
        self.havemoved = False
        self.showstep = False
        self.show_heatmap = False
        self.show_analyse = False
        self.heatmaps=[]
        self.heatmap_texts=[]
        self.var_name1 = StringVar()
        self.var_name2 = StringVar()
        self.current_stones = np.zeros([go.N, go.N], dtype=np.int8)
        self.step = 0
        self.change_board = None
        self.change_step = False
        self.analyser = None
        self.pause_analyse = True
        self.canvas = None
        self.blackimg = None
        self.whiteimg = None
        self.hqh = None
        self.bqh = None
        self.initboard()
        self.resize(0,0,self.screen_width, self.screen_height)
        #self.resize(0,0,self.screen_width, self.screen_height)
        sys.stdout = self.gtpinfo
        sys.stderr = self.info
        # sys.stdin = self.gtpinfo

    def on_resize(self,event):
        #self.width = event.width
        self.height = event.height
        # resize the canvas 
        minwh = event.height
        if abs(event.width-event.height)>10:
            minwh = min(event.width, event.height)
            self.canvas.config(width=minwh, height=minwh)
            return
        self.canvas_resize(event.x, event.y, minwh, minwh)

    def resize(self, left, top, width, height):
        self.root.geometry('%dx%d+%d+%d' % (width, height, left, top))

    def canvas_resize(self, left, top, width, height):
        if self.canvas:
            self.canvas.delete(ALL)
        self.current_stones=np.zeros([go.N, go.N], dtype=np.int8)
        self.height = height
        #self.current_stones = np.zeros([go.N, go.N], dtype=np.int8)
        #self.board = None
        hpad = 0
        self.linespace = int((self.height-hpad) / (go.N+1))
        self.xs = int(self.linespace * 1)
        self.ys = int(self.linespace * 1)
        img=self.img1.resize((int(self.height), int(self.height)), Image.ANTIALIAS)
        self.bjimg=ImageTk.PhotoImage(img)
        img=self.img2.resize((int(self.linespace*0.95), int(self.linespace*0.95)), Image.ANTIALIAS)
        self.blackstone = ImageTk.PhotoImage(img)
        img=self.img3.resize((int(self.linespace*0.95), int(self.linespace*0.95)), Image.ANTIALIAS)
        self.whitestone = ImageTk.PhotoImage(img)
        img=self.img4.resize((int(self.height*0.15), int(self.height*0.15)), Image.ANTIALIAS)
        self.hqh = ImageTk.PhotoImage(img)
        if self.blackimg:
            self.blackimg.config(image = self.hqh)
        img=self.img5.resize((int(self.height*0.15), int(self.height*0.15)), Image.ANTIALIAS)
        self.bqh = ImageTk.PhotoImage(img)
        if self.whiteimg:
            self.whiteimg.config(image = self.bqh)
        img=self.img6.resize((int(self.linespace*0.34), int(self.linespace*0.34)), Image.ANTIALIAS)
        self.last = ImageTk.PhotoImage(img)
        self.drawboard()
        #self.canvas.addtag_all("all")

    def initboard(self):
        font = ("宋体", 10) #scalecanvas.Scale
        self.canvas = Canvas(self.root, bg='gray', width=self.height, height=self.height)
        self.stones = np.array([([None] * go.N) for _ in (range(go.N))])
        self.steptexts = {}  #, log="/logs/info.log", log="/logs/important.log"
        self.info = ScrollText(self, self.root, width=75, font=font, padx=int(self.height * 0.004), pady=int(self.height * 0.002))
        self.gtpinfo = ScrollText(self, self.root, width=75, font=font, padx=int(self.height * 0.004), pady=int(self.height * 0.002))
        self.cmd = Entry(self.root, width=75, font=font)
        self.status = Label(self.root, text='盘面状态：', font=font, anchor=W)
        self.blackimg = Label(self.root, anchor=CENTER, image=self.hqh)
        self.player1_name = Label(self.root, width=16, text='岁的时候看', font=font, anchor=CENTER)
        self.player1_eat = Label(self.root, width=16, text='eat1', font=font, anchor=CENTER)
        self.player1_winr = Label(self.root, width=16, text='胜率：', font=font, anchor=CENTER)
        self.whiteimg = Label(self.root, anchor=CENTER, image=self.bqh)
        self.player2_name = Label(self.root, width=16, text='pp', font=font, anchor=CENTER)
        self.player2_eat = Label(self.root, width=16, text='eat2', font=font, anchor=CENTER)
        self.player2_winr= Label(self.root, width=16, text='胜率：', font=font, anchor=CENTER)
        cmdlabel = Label(self.root, width=16, text='命令行：', font=font, anchor=E)
        klb = Frame(self.root, height=int(self.height * 0.02))
        
        # grid
        padx = int(self.height * 0.004)
        pady = int(self.height * 0.002)
        padyb = self.height*0.006
        self.canvas.grid(row=0, column=0, rowspan=9, sticky=NSEW)
        self.blackimg.grid(row=0, column=1, stick=EW, padx=padx, pady=pady)
        self.player1_name.grid(row=1, column=1, stick=N, padx=padx, pady=pady)
        self.player1_eat.grid(row=2, column=1, stick=N, padx=padx, pady=pady)
        self.player1_winr.grid(row=3, column=1, stick=N, padx=padx, pady=pady)
        klb.grid(row=4, column=1, stick=NS)
        self.player2_winr.grid(row=5, column=1, stick=N, padx=padx, pady=pady)
        self.player2_eat.grid(row=6, column=1, stick=N, padx=padx, pady=pady)
        self.player2_name.grid(row=7, column=1, stick=N, padx=padx, pady=pady)
        self.whiteimg.grid(row=8, column=1, stick=EW, padx=padx, pady=pady)
        self.info.grid(row=0, column=2, rowspan=4, sticky=NS)
        self.gtpinfo.grid(row=5, column=2, rowspan=4, sticky=NS)
        self.status.grid(row=9, column=0, sticky=EW, padx=padx, pady=padyb)
        cmdlabel.grid(row=9, column=1, sticky=EW, padx=padx, pady=padyb)
        self.cmd.grid(row=9, column=2, sticky=EW, padx=padx, pady=padyb)
        #self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)
        #self.root.rowconfigure(4, weight=1)

        self.build_menu()
        self.scores = []
        self.last_img = None
        self.last_text = None
        self.root.bind('<Key>', self.on_key)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_undo)
        self.canvas.bind('<Control-Button-1>', self.on_ctl_click)
        self.canvas.bind("<Configure>", self.on_resize)

    def drawboard(self):
        font = ("宋体", 12)
        lw = int(self.linespace*0.01)
        dotr = int(self.linespace*0.08)
        self.canvas.create_image(int(self.bjimg.width() / 2.0),
                                 int(self.bjimg.height() / 2.0), image=self.bjimg)
        for i in range(go.N):
            self.canvas.create_text(i * self.linespace + self.xs, int(self.linespace*0.3),
                                    font=font, text=go.COLLUM_STR[i])
            self.canvas.create_text(i * self.linespace + self.xs, self.ys + int((go.N-0.3) * self.linespace),
                                    font=font, text=go.COLLUM_STR[i])
            self.canvas.create_line(i * self.linespace + self.xs, self.ys,
                                    i * self.linespace + self.xs, (go.N - 1) * self.linespace + self.ys, width=self.linespace*0.01)
            if i == 0 :
                self.canvas.create_line(self.xs-self.linespace*0.06 , self.ys - self.linespace*0.08 ,
                                        self.xs - self.linespace * 0.06,
                                        (go.N - 0.92) * self.linespace + self.ys, width=self.linespace*0.04)
            if i == go.N - 1:
                self.canvas.create_line((i+0.06) * self.linespace + self.xs, self.ys - self.linespace*0.08,
                                        (i + 0.06) * self.linespace + self.xs,
                                        (go.N - 0.92) * self.linespace + self.ys , width=self.linespace*0.04)
        for j in range(go.N):
            self.canvas.create_text(int(self.linespace*0.3), self.ys + j * self.linespace,
                                    font=font, text=str(go.N - j))
            self.canvas.create_text(self.xs + (go.N - 0.3) * self.linespace , self.ys + j * self.linespace,
                                    font=font, text=str(go.N - j))
            self.canvas.create_line(self.xs, j * self.linespace + self.ys,
                                    (go.N - 1) * self.linespace + self.xs, j * self.linespace + self.ys, width=self.linespace*0.01)
            if j == 0:
                self.canvas.create_line(self.xs -self.linespace*0.08, self.ys-self.linespace*0.06,
                                        (go.N - 0.92) * self.linespace + self.xs,
                                        self.ys - self.linespace * 0.06, width=self.linespace*0.04)
            if j == go.N - 1:
                self.canvas.create_line(self.xs - self.linespace * 0.08,
                                        (j+0.06) * self.linespace + self.ys,
                                        (go.N - 0.92) * self.linespace + self.xs ,
                                        (j+0.06) * self.linespace + self.ys, width=self.linespace*0.04)
        dots = list(map(lambda x: x * 6 + 3, range(int(go.N / 6))))
        for i in dots:
            for j in dots:
                self.canvas.create_oval(i * self.linespace + self.xs - dotr, j * self.linespace + self.ys - dotr,
                                        i * self.linespace + self.xs + dotr, j * self.linespace + self.ys + dotr,
                                        fill='black')

    def build_menu(self):
        # 在大窗口下定义一个菜单实例
        font = ("宋体", 12)
        menubar = Menu(self.root, font=font)
        # 在顶级菜单实例下创建子菜单实例, 去掉虚线tearoff=0
        fmenu = Menu(menubar, font=font, tearoff=0)
        # 为每个子菜单实例添加菜单项
        for lbl, cmd in zip(['打开', '保存'], [self.open, self.save]):
            fmenu.add_command(label=lbl, command=cmd)
        # 给子菜单添加分割线
        fmenu.add_separator()
        # 继续给子菜单添加菜单项
        for each in zip(['自动打谱', '手动打谱'],
                        [None, None]):
            fmenu.add_radiobutton(label=each)
        fmenu.add_separator()
        for lbl, cmd, acc in zip(['下一步', '上一步'],
                                 [self.next_step, self.previous_step], ['>', '<']):
            fmenu.add_radiobutton(label=lbl, command=cmd, accelerator=acc)
        fmenu.add_separator()
        fmenu.add_command(label='退出', command=quit)

        funcmenu = Menu(menubar, font=font, tearoff=0)
        for lbl, cmd in zip(['对战', '数据', '训练', '获取'],
                            [self.play, self.get_traindata, self.train, self.get_board]):
            funcmenu.add_command(label=lbl, command=cmd)
        funcmenu.add_separator()
        funcmenu.add_command(label='停止训练', command=self.stoptrain)

        vmenu = Menu(menubar, font=font, tearoff=0)
        for lbl, cmd in zip(['形势判断', '关闭形势判断', '指点', '试下'],
                            [self.show_score, self.hidden_score, None, None]):
            vmenu.add_command(label=lbl, command=cmd)
        vmenu.add_separator()
        for lbl, cmd in zip(['悔棋', '不走', '认输', '重新开始'],
                            [self.undo, self.move_pass, self.resign, self.restart]):
            vmenu.add_command(label=lbl, command=cmd)

        emenu = Menu(menubar, font=font, tearoff=0)
        for lbl, cmd in zip(['参数设置', '棋盘设置', '棋手设置'],
                            [self.cfg_window, None, None]):
            emenu.add_command(label=lbl, command=cmd)
        emenu.add_separator()
        emenu.add_command(label="降低学习率", command = self.reduce_lr)
        self.chkstep = tkinter.BooleanVar()
        self.chkheatmap = tkinter.BooleanVar()
        self.chkanalyse = tkinter.BooleanVar()
        emenu.add_checkbutton(label='显示手数', variable=self.chkstep, command=self.change_show_step)
        emenu.add_checkbutton(label='显示热图', variable=self.chkheatmap, command=self.change_show_heatmap)
        emenu.add_checkbutton(label='分析模式', variable=self.chkanalyse, command=self.change_analyse)
        policymenu = Menu(emenu, font=font)
        self.cfg = tkinter.StringVar()
        for lbl, cfg in zip(['AlphgoZero', '5层2特征', '5层9特征', '6层2特征', '10层9特征', '20层2特征', '6层17特征'],
                            ['config-az', 'config-b5f2', 'config-b5f9', 'config-b6f2', 'config-b10f9', 'config-b20f2',
                             'config-b6f17']):
            policymenu.add_radiobutton(label=lbl, variable=self.cfg, value=cfg)
        self.cfg.set(config.last_config)
        emenu.add_cascade(label='网络配置', menu=policymenu)
        emenu.add_separator()
        playermenu1 = Menu(emenu, font=font)
        playermenu2 = Menu(emenu, font=font)
        for each in ['人', 'Policy', 'Ramdom', 'MTCS']:
            playermenu1.add_radiobutton(label=each, variable=self.var_name1, value=each)
            playermenu2.add_radiobutton(label=each, variable=self.var_name2, value=each)
        for name, cmd in config.apps.items():
            playermenu1.add_radiobutton(label=name, variable=self.var_name1, value=name)
            playermenu2.add_radiobutton(label=name, variable=self.var_name2, value=name)

        self.var_name1.set("人")
        self.var_name2.set("人")
        emenu.add_cascade(label='棋手1', menu=playermenu1)
        emenu.add_cascade(label='棋手2', menu=playermenu2)
        emenu.add_separator()
        self.summary = tkinter.BooleanVar()
        emenu.add_checkbutton(label='打开观察表', variable=self.summary, command=self.change_summary)
        self.summary.set(config.summary)

        amenu = Menu(menubar, font=font, tearoff=0)
        for lbl, cmd in zip(['测试', '版权信息', '关于'],
                            [self.test, None, None]):
            amenu.add_command(label=lbl, command=cmd)
        # 为顶级菜单实例添加菜单，并级联相应的子菜单实例
        menubar.add_cascade(label='棋谱', menu=fmenu)
        menubar.add_cascade(label='功能', menu=funcmenu)
        menubar.add_cascade(label='操作', menu=vmenu)
        menubar.add_cascade(label='设置', menu=emenu)
        menubar.add_cascade(label='帮助', menu=amenu)
        # 菜单实例应用到大窗口中
        self.root['menu'] = menubar

    def get_label(self):
        ml=int(go.N/2)
        return [go.N/go.MAX_BOARD, self.linespace/self.screen_height, 
                (ml * self.linespace + self.xs)/self.screen_height,
                (ml * self.linespace + self.ys)/self.screen_height]

    def change_show_step(self):
        self.showstep = self.chkstep.get()
        self.update()

    def change_show_heatmap(self):
        self.show_heatmap = self.chkheatmap.get()

    def change_analyse(self):
        self.show_analyse = self.chkanalyse.get()
        if self.show_analyse:
            self.show_heatmap = True
            self.chkheatmap.set(True)
            cmd =config.apps["ZiGo"].split(",")
            self.analyser = intergtp.GTP_player(cmd, "ZiGo", go.WHITE)
            #self.analyser.stop_analyse()
            #if self.board and self.board.step>0:
            #    sgf = sgfparser.SgfParser(board=self.board)
            #    sgf.save("temp.sgf")
            #    self.analyser.analyse(os.path.curdir + "/temp.sgf")
            #else:
            self.analyser.analyse()
        elif self.analyser:
            self.analyser.stop_analyse()
            time.sleep(0.6)
            self.analyser.quit()
            self.analyser = None
            self.show_heatmap = False
            self.chkheatmap.set(self.show_heatmap)
            if os.path.exists("temp.sgf"):
                os.remove("temp.sgf")

    def analyse(self):
        self.pause_analyse = not self.pause_analyse
        if self.analyser:
            self.analyser.send("pause_analyse")

    def start(self, pos=None, p1name=None, p2name=None):
        self.clear()
        if pos is None:
            self.board = board.Board()
        else:
            self.board = pos
            if not p1name:
                p1name = pos.player1_name
            if not p2name:
                p2name = pos.player2_name
            self.player1_name['text'] = p1name
            self.player2_name['text'] = p2name
        pos = self.board.get_board(self.step)
        self.update(pos)

    def update(self, pos=None):
        if pos is None:
            pos = self.board
        if not pos:
            return

        self.step = pos.step
        changed = np.where(self.current_stones != pos.stones)
        for i in range(len(changed[0])):
            coor = changed[0][i], changed[1][i]
            if coor == go.PASS:
                continue
            if self.current_stones[coor] != go.EMPTY:
                self.pickup(coor)
            else:
                self.downstone(pos.stones[coor], coor)
        self.current_stones = pos.stones.copy()

        if self.step < 1:
            if self.last_text:
                self.canvas.delete(self.last_text)
            if self.last_img:
                self.canvas.delete(self.last_img)
            return
        move = pos.recent[self.step - 1].move
        c = pos.recent[self.step - 1].color
        font = ("黑体", int(12* self.linespace/80))
        y, x = move[0], move[1]
        if self.showstep:
            if not self.steptexts:
                if self.last_text:
                    self.canvas.delete(self.last_text)
                    self.last_text = None
                self.steptexts = {}
                for i,pm in enumerate(pos.recent):
                    m = pm.move
                    self.steptexts[m] = (self.canvas.create_text(m[1] * self.linespace + self.xs,
                                        m[0] * self.linespace + self.ys, text=str(i+1),
                                        font=font, fill='white' if pm.color == go.BLACK else 'black'))
            else:
                for i in range(len(changed[0])):
                    coor = changed[0][i], changed[1][i]
                    if coor == go.PASS:
                        continue
                    step = pos.get_step_from_move(coor)
                    if self.current_stones[coor] == go.EMPTY:
                        self.canvas.delete(self.steptexts[coor])
                    else:
                        color = pos.get_color(step)
                        self.steptexts[coor] = (self.canvas.create_text(coor[1] * self.linespace + self.xs,
                                                coor[0] * self.linespace + self.ys, text=str(step+1),
                                                font=font, fill='white' if color == go.BLACK else 'black'))
        else:
            if self.steptexts:
                for m,t in self.steptexts.items():
                    self.canvas.delete(t)
                self.steptexts.clear()
            if self.last_text:
                self.canvas.delete(self.last_text)
            self.last_text = self.canvas.create_text(x * self.linespace + self.xs,
                                                     y * self.linespace + self.ys, text=str(self.step),
                                                     font=font, fill='white' if c == 1 else 'black')

        if self.last_img:
            self.canvas.delete(self.last_img)
        self.last_img = self.canvas.create_image((x + 0.3) * self.linespace + self.xs,
                                                 (y + 0.3) * self.linespace + self.ys, image=self.last)

        score = pos.score()
        self.player1_eat['text'] = '吃子：%d' % (pos.caps[0])
        self.player2_eat['text'] = '吃子：%d' % (pos.caps[1])
        wb = pos.points
        if self.show_heatmap and self.info.values:
            m = list(self.info.values.keys())[0]
            wb = self.info.values[m].winrate
            #self.update_heatmap(self.info.values)
        elif not self.show_heatmap:
            if self.heatmaps:
                for hm in self.heatmaps:
                    self.canvas.delete(hm)
                self.heatmaps = []
            if self.heatmap_texts:
                for ht in self.heatmap_texts:
                    self.canvas.delete(ht)
                self.heatmap_texts=[]
            
        wb = wb if c==go.BLACK else -wb
        ww = -wb
        #self.status['text'] = '{}手，估计黑方胜率：{:.1f}，白方胜率：{:.1f}，{}领先{:.1f}子。'.format(pos.step,\
        #                                                       wb, ww, go.get_color_str(score), abs(score))
        #self.info.see(END)
        if c==go.BLACK:
            self.player1_winr["text"] = "黑：{}".format(go.get_point_str(wb))
        else:
            self.player2_winr["text"] = "白：{}".format(go.get_point_str(ww))

    def update_heatmap(self, values):
        if self.show_heatmap:
            if self.heatmaps:
                for hm in self.heatmaps:
                    self.canvas.delete(hm)
                self.heatmaps = []
            if self.heatmap_texts:
                for ht in self.heatmap_texts:
                    self.canvas.delete(ht)
                self.heatmap_texts=[]
            #vs={}
            font = ("宋体", int(8* self.linespace/80))
            r = self.linespace*0.5
            #for m,v in values.items():
                #m = utils.unflatten_coords(i)
                #if self.current_stones[m] == go.EMPTY and v>0:
                #    vs[m] = values[i]
            allvisits=0
            for m,v in values.items():
                allvisits += v.visits
            j = 0
            for m,v in values.items():
                j += 1
                if j>10:
                    break
                vm = int((v.visits/allvisits)*164)
                red = 255
                g = min(164, max(0, 164-vm))
                b = min(164, max(0, 164-vm))
                color ='#%02X%02X%02X' %(red, g, b) 
                hm = self.canvas.create_oval(m[1] * self.linespace + self.xs - r, m[0] * self.linespace + self.ys - r,
                                        m[1] * self.linespace + self.xs + r, m[0] * self.linespace + self.ys + r,
                                        fill=color)
                self.heatmaps.append(hm)
                ht = self.canvas.create_text(m[1] * self.linespace + self.xs,
                                m[0] * self.linespace + self.ys - int(r/3), text=str("{:.1f}".format(v.winrate)),
                                font=font, fill='black')
                self.heatmap_texts.append(ht)
                ht2 = self.canvas.create_text(m[1] * self.linespace + self.xs,
                                m[0] * self.linespace + self.ys+int(r/3), text=str(v.visits),
                                font=font, fill='black')
                self.heatmap_texts.append(ht2)


    def clear(self):
        self.canvas.delete(ALL)
        self.drawboard()
        self.current_stones = np.zeros([go.N, go.N], dtype=np.int8)
        self.board = None

    def show(self, pos, next_move=None):
        self.clear()
        self.showstep = True
        for i, pm in enumerate(pos.recent):
            self.downstone(pm.color, pm.move)
            if pm.captured:
                for p in pm.captured:
                    self.pickup(p)
        if next_move:
            self.downstone(pos.to_move, next_move)

    def on_click(self, event):
        if self.change_board:
            self.board=self.change_board
            self.showstep = self.change_step
            self.change_board = None
            self.update()
        col, row = self.coortoline(event.x, event.y)
        if col < 0 or col > go.N - 1 or row < 0 or row > go.N - 1:
            return
        self.clicked = (row, col)
        if not self.board:
            self.board=board.Board()
        if self.show_analyse and self.analyser:
            c = self.board.to_move
            self.board.play_move(self.clicked)
            vertext =go.get_cmd_from_coor(self.clicked)
            self.analyser.play(vertext, c)
            self.update()
            #if not self.pause_analyse:
            #    self.analyser.analyse()

    def on_undo(self, event):
        self.undo()

    def downstone(self, c, move):
        y, x = move[0], move[1]
        if x not in range(go.N) or y not in range(go.N):
            return
        tag = go.get_cmd_from_coor(move)
        # print('第%d手下在了：%s' % (self.step, tag))
        #lwh = int(self.linespace * 0.025)
        self.stones[move] = self.canvas.create_image(x * self.linespace + self.xs, y * self.linespace + self.ys,
                                                     image=self.blackstone if c == go.BLACK else self.whitestone,
                                                     tags=(tag))
        '''
        if self.last_move:
            self.canvas.delete(self.last_move)
        self.last_move = self.canvas.create_rectangle(i*self.linespace+self.xs-lwh,
            j*self.linespace+self.ys-lwh, i*self.linespace+self.xs+lwh,
            j*self.linespace+self.ys+lwh, fill='white' if c==1 else 'black')
        '''

    def pickup(self, coor):
        tag = go.get_cmd_from_coor(coor)
        # print('提子：%s' % (tag))
        if self.stones[coor]:
            self.canvas.delete(self.stones[coor])
        #if self.steptexts[coor]:
        #    self.canvas.delete(self.steptexts[coor])

    def show_message(self, msg=None, status=None):
        if msg:
            # if self.message:
            #    self.canvas.delete(self.message)
            # self.message = self.canvas.create_text(360, 650, text=msg)
            #txt = self.info.get(0.0, END)
            #if txt.count('\n') > 200:
            #    self.info.delete(0.0, 20.0)
            self.info.insert(END, msg + '\n')
            self.info.see(END)
        if status:
            self.status['text'] = status

    def hidden_score(self):
        if len(self.scores) > 0:
            for rect in self.scores:
                self.canvas.delete(rect)

    def show_score(self, pos=None):
        if pos:
            self.board = pos
        if not self.board:
            return 0
        score = self.board.score()
        self.hidden_score()
        lwh = int(self.height * 0.005)
        for b in self.board.scoreb:
            self.scores.append(self.canvas.create_rectangle(b[0] * self.linespace + self.xs - lwh,
                                                            b[1] * self.linespace + self.ys - lwh,
                                                            b[0] * self.linespace + self.xs + lwh,
                                                            b[1] * self.linespace + self.ys + lwh, fill='black'))
        for b in self.board.scorew:
            self.scores.append(self.canvas.create_rectangle(b[0] * self.linespace + self.xs - lwh,
                                                            b[1] * self.linespace + self.ys - lwh,
                                                            b[0] * self.linespace + self.xs + lwh,
                                                            b[1] * self.linespace + self.ys + lwh, fill='white'))
        return score

    def coortoline(self, x, y):
        lx = int((x - (self.xs - self.linespace / 2.0)) / self.linespace)
        ly = int((y - (self.ys - self.linespace / 2.0)) / self.linespace)
        return lx, ly

    # 菜单命令
    def quit(self):
        if self.analyser:
            self.analyser.stop_analyse()
            self.analyser.quit()
            self.analyser=None
        if self.info.log:
            self.info.save_log()
        if self.gtpinfo.log:
            self.gtpinfo.save_log()
        config.running = False
        time.sleep(0.5)
        sys.stdout = None
        sys.stderr = None
        self.root.quit()
        self.root.destroy()
        exit()

    def on_key(self, event):
        if event.keysym == 'Next' or event.keysym == 'Right':
            self.next_step()
        elif event.keysym == 'Prior' or event.keysym == 'Left':
            self.previous_step()
        elif event.keysym == 'space':
            self.analyse()
        elif event.keysym == 'p' or event.keysym == 'P':
            self.move_pass()
        elif event.keysym == 'Escape':
            if self.change_board:
                self.board=self.change_board
                self.showstep = self.change_step
                self.change_board = None
                self.update()


    def on_ctl_click(self, event):
        if self.change_board:
            self.board=self.change_board
            self.change_board = None
            self.showstep = self.change_step
            self.update()
        col, row = self.coortoline(event.x, event.y)
        if col < 0 or col > go.N - 1 or row < 0 or row > go.N - 1:
            return
        if (row,col) in self.info.values.keys():
            self.change_step = self.showstep
            self.showstep = False
            self.update()
            pos = self.board.copy()
            self.board.step = 0
            self.board.recent = []
            self.showstep=True
            for move in self.info.values[(row,col)].nextmoves.split(" "):
                m = go.get_coor_from_gtp(move)
                if m==go.PASS or m==go.RESIGN or m is None:
                    continue
                self.board.play_move(m)
            self.update()
            self.change_board = pos

    def open(self):
        fn = tkinter.filedialog.askopenfilename(filetypes=[("sgf格式", "sgf")])
        pos = sgfparser.get_sgf_board(fn)
        self.start(pos)

    def save(self):
        fn = tkinter.filedialog.asksaveasfilename(filetypes=[("sgf格式", "sgf")])
        sgf = sgfparser.SgfParser(board=self.board)
        sgf.save(fn)
        

    def next_step(self):
        if self.step >= self.board.step:
            return
        if self.analyser:
            c = self.board.get_color(self.step)
            move = self.board.recent[self.step].move
            vertex = go.get_cmd_from_coor(move)
            self.analyser.play(vertex, c)
        self.step += 1
        pos = self.board.get_board(self.step)
        self.update(pos)

    def previous_step(self):
        if self.step < 1:
            return
        if self.analyser:
            self.analyser.send("undo")
        self.step -= 1
        pos = self.board.get_board(self.step)
        self.update(pos)

    def undo(self):
        self.clicked = (-3, 0)
        if self.show_analyse and self.analyser:
            c = self.board.to_move
            self.board.play_move(self.clicked)
            self.analyser.send("undo")
            self.update()

    def move_pass(self):
        self.clicked = go.PASS
        if self.show_analyse and self.analyser:
            c = self.board.to_move
            self.board.play_move(self.clicked)
            vertext =go.get_cmd_from_coor(self.clicked)
            self.analyser.play(vertext, c)
            self.update()

    def resign(self):
        self.clicked = go.RESIGN

    def play(self):
        #config.read_cfg(self.cfg.get())
        #go.set_board_size(config.board_size)
        main.play(player1=self.var_name1.get(), player2=self.var_name2.get(), qp=self, gtpon=True)

    def train(self):
        self.show_message('开始训练……')
        datas = dataset.DataSet(self)
        datas.start_load()
        self.train_net = network.Network(self.screen_height, is_train=True)
        self.show_message(status='正在训练……')
        trainth = threading.Thread(target = self.train_net.train, args=(datas,))
        trainth.setDaemon(True)
        trainth.start()

    def get_board(self):
        #path = tkinter.filedialog.askdirectory()
        self.show_message('开始获取棋谱%s中的数据……')
        datas = dataset.DataSet(self)
        plane = datas.get_plane()
        #print(plane.shape())
        #self.get_label()
        #self.train_net = network.Network(self.screen_height, is_train=False)
        #vs = self.train_net.run(plane)
        #print(vs)
        #testoval = self.canvas.create_oval(vs[2]-20, vs[3]-20,vs[2]+20,vs[3]+20)
        print("board size:{}, line space:{}, x:{}, y:{}".format(vs[0], vs[1], vs[2], vs[3]))

    def reduce_lr(self):
        self.self_train.reduce_lr()

    def restart(self):
        if self.analyser:
            self.analyser.send("clear_board")
        self.start()

    def get_traindata(self):
        #config.read_cfg(self.cfg.get())
        #go.set_board_size(config.board_size)
        #path = tkinter.filedialog.askdirectory()
        self.show_message('开始自我训练……')
        datas = dataset.DataSet(self)
        datas.start_auto()
        #self.train_net = network.Network(self.screen_height, is_train=True)
        self.show_message(status='正在进行增强学习……')
        #trainth = threading.Thread(target = self.train_net.train, args=(datas,))
        #trainth.setDaemon(True)
        #trainth.start()

    def cfg_window(self):
        cfg = config.ConfigWindow(self.height*1, self.height*0.6)
        cfg.mainloop()

    def change_summary(self):
        config.summary = self.summary.get()
        if config.summary:
            if get_process_count('tensorboard.exe') == 0:
                subprocess.Popen(["tensorboard.exe", "--logdir",
                                  "d:\\myprogram\\ai\\zigoAc\\log"], shell=False)
            # C:\Program Files (x86)\Google\Chrome\Application\chrome.exe
            subprocess.Popen(["C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "http://localhost:6006"])
        else:
            if get_process_count('tensorboard.exe') > 0:
                subprocess.run(["taskkill", "/f", "/t", "/im", "tensorboard.exe"])

    def stoptrain(self):
        config.running = False

    def test(self):
        config.read_cfg(self.cfg.get())
        go.set_board_size(config.board_size)
        selfplay.selftest(qp=self)
        # selfplay.elo()
