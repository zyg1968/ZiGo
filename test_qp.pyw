from tkinter import *
import scrolltext
from qipan import QiPan
import sys
from PIL import Image, ImageTk
import threading
import go
import time
import config

def gui():
    root=Tk()
    def on_quit():
        config.running = False
        sys.stdout=None
        root.quit()
        root.destroy()
        exit()

    root.protocol("WM_DELETE_WINDOW", on_quit)
    #设置窗口图标
    root.iconbitmap('pic/zigo.ico')
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight() 
    width_mm = root.winfo_screenmmwidth()
    height_mm = root.winfo_screenmmheight() 
    # 2.54 cm = in
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    width_dpi = width_px/width_in
    height_dpi = height_px/height_in 
    width = int(width_dpi * 14)
    height = int(height_dpi * 7.98)
    root.geometry('%dx%d+%d+%d' % (width, height, 0, 0))
    root.resizable(width=True, height=True)
    root.title("围棋")
    img=Image.open('pic/bj3.png')
    img=img.resize((int(height*1.4), int(height*1.0)), Image.ANTIALIAS)
    bjimg=ImageTk.PhotoImage(img)
    img=Image.open('pic/Black61.png')
    img=img.resize((int(width_dpi*0.31), int(height_dpi*0.31)), Image.ANTIALIAS)
    blackstone = ImageTk.PhotoImage(img)
    img=Image.open('pic/White61.png')
    img=img.resize((int(width_dpi*0.31), int(height_dpi*0.31)), Image.ANTIALIAS)
    whitestone = ImageTk.PhotoImage(img)
    img=Image.open('pic/hqh.png')
    img=img.resize((int(width_dpi*0.91), int(height_dpi*0.86)), Image.ANTIALIAS)
    hqh = ImageTk.PhotoImage(img)
    img=Image.open('pic/bqh.png')
    img=img.resize((int(width_dpi*0.91), int(height_dpi*0.86)), Image.ANTIALIAS)
    bqh = ImageTk.PhotoImage(img)
    img=Image.open('pic/last.png')
    img=img.resize((int(width_dpi*0.11), int(height_dpi*0.11)), Image.ANTIALIAS)
    last = ImageTk.PhotoImage(img)
    qp=QiPan('显示', root, bjimg, blackstone, whitestone, hqh, bqh, last, width_dpi, height_dpi)
    return qp, root

class PlayThread (threading.Thread):
    def __init__(self, name, qp, read_file=None, gtpon=False):
        threading.Thread.__init__(self)
        self.name = name
        self.qp=qp
        self.running=False
        self.gtpon=gtpon
        self.gomanager=go.Board(to_play=go.BLACK)
        self.gomanager.player1_name = 'Player1'
        self.gomanager.player2_name = 'Player2'
        self.qp.start(self.gomanager)

    def run(self):
        print ("开启线程： " + self.name)
        self.running=True
        self.playloop()
        self.running=False
        print ("退出线程： " + self.name)

    def playloop(self):
        passed=0
        while(self.running):
            while(not self.qp.clicked and not self.qp.havemoved):
                time.sleep(0.1)
            c = self.gomanager.to_play
            if not self.qp.havemoved:
                coor=self.qp.clicked
                self.qp.clicked = None
                vertex = go.get_vertex_from_coor(coor)
            if coor is None or coor==go.PASS:
                if c == go.BLACK:
                    passed |= 1
                else:
                    passed |= 2
                print('%s pass了。' % (player.name))
                if passed==3:
                    break
            pstr = go.get_cmd_from_vertex(vertex)
            print('%s 准备走：%s(%d,%d)' % (go.get_color_str(c), pstr, coor[0], coor[1]))
            ill, caps = self.gomanager.play_move(coor, c)
            if ill>0:
                print('%s方%s着法不合法，因为%s。' % (go.get_color_str(c), 
                    pstr, go.get_legal_str(ill)))
            self.qp.update(self.gomanager)
            self.qp.havemoved = False
            self.running=config.running
            time.sleep(0.1)
        msg = "对局结束。" + go.result_str(self.gomanager.result)
        print(msg)
        self.gomanager.show_message(msg=msg, status=msg)

    def set_size(self, n):
        go.set_board_size(n)
        self.gomanager.clear()

    def set_komi(self, komi):
        self.gomanager.komi = komi

    def clear(self):
        self.gomanager = go.Board(komi=7.5)

    def make_move(self, color, vertex):
        coords = self.gomanager.get_coor_from_vertex(vertex)
        ill, caps = self.gomanager.play_move(coords, color=translate_gtp_colors(color))
        if ill>0:
            return False
        return True

    def get_move(self, color):
        pass


if __name__ == '__main__':
    qp, root=gui()
    config.running = True
    thplay = PlayThread('下棋', qp)
    thplay.setDaemon(True)
    thplay.start()
    root.mainloop()
