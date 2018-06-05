#!/usr/bin/env python3
#
#    This file is part of ZiGo.
#    Copyright (C) 2018 ZiGo
#
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk

class CGOSWindow():
    def __init__(self, hpx):
        self.top = Toplevel()
        self.gui(hpx)
        self.closed = False
        self.top.wm_attributes("-topmost", 1)
        #self.top.mainloop()

    def gui(self, hpx):
        xs = hpx/1800
        width = int(hpx * 1.2)
        height = int(hpx * 0.8)
        padx=int(hpx*0.005)
        pady=int(hpx*0.005)
        self.top.geometry('%dx%d+%d+%d' % (width, height, hpx* 0.2, hpx*0.1))
        self.top.resizable(width=True, height=True)
        self.top.title("CGOS比赛")
        #self.table = TableView(self.top,rows=1, columns=6)
        style = ttk.Style(self.top)
        style.configure('Treeview', rowheight=int(50*xs))  #SOLUTION
        self.table = ttk.Treeview(self.top, show="headings", height=int(50*xs), 
                     columns=("a", "b", "c", "d", "e", "f"))
        self.table.column('a', minwidth=90, width=int(170*xs), stretch=NO, anchor='center')
        self.table.column('b', minwidth=120, width=int(230*xs), stretch=NO, anchor='center')
        self.table.column('c', minwidth=80, width=int(120*xs), stretch=NO,  anchor='center')
        self.table.column('d', minwidth=180, width=int(220*xs), anchor='center')
        self.table.column('e', minwidth=180, width=int(220*xs), anchor='center')
        self.table.column('f', minwidth=100, width=int(220*xs), stretch=NO, anchor='center')
        self.table.heading('a', text='编号')
        self.table.heading('b', text='日期')
        self.table.heading('c', text='时间')
        self.table.heading('d', text='白方')
        self.table.heading('e', text='黑方')
        self.table.heading('f', text='结果')        
        self.table.column('#5', stretch=True)
        self.table.tag_configure('a', font='Arial 20')
        self.table.tag_configure('oddrow', background='#eeeeff')
        self.vbar = ttk.Scrollbar(self.top, orient=VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=self.vbar.set)
        self.table.grid(row = 0, column = 0,sticky=NSEW, padx=padx, pady=pady)
        self.vbar.grid(row=0,column=1,sticky=NS)
        self.top.columnconfigure(0, weight = 1)
        self.top.rowconfigure(0, weight = 1)
        self.table.bind('<Double-Button-1>', self.on_dbl_click)

    def on_dbl_click(self, event):
        item = self.table.selection()[0]
        itemid = self.table.item(item, "values")[0]
        self.hide()
        self.closed = True
        self.show_match(itemid)

    def hide(self):
        self.top.withdraw()

    def show(self):
        self.top.update()
        self.top.deiconify()
        self.closed = False

    def cancel(self):
        self.top.destroy()

    def add(self, values, show_match):
        #[id,date,time,board_size,komi,white,black,result]
        #404962	Zen-11.4n-1c(2936)	leela-0.11.0-p1600(2960) 404962	Zen-11.4n-1c(2936)	—	leela-0.11.0-p1600(2960)
        self.show_match=show_match
        items = self.table.get_children()
        count = len(items)
        self.table.insert('',count, values=values)
        if count % 2==0:
            self.table.item(self.table.get_children()[count], tags=('oddrow'))

class TableView(Frame):
    def __init__(self, parent, rows=1, columns=1):
        # use black background so it "peeks through" to 
        # form grid lines
        self.padx = 1
        self.pady = 1
        self.rows=rows
        self.columns=self.columns
        self.column_widths = []
        Frame.__init__(self, parent, background="black")
        self._widgets = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                self.column_widths.append(10)
                label = Label(self, text="%s/%s" % (row, column), 
                                 borderwidth=0, width=10)
                label.grid(row=row, column=column, sticky="nsew", padx=1, pady=1)
                current_row.append(label)
            self._widgets.append(current_row)

        for column in range(columns):
            self.grid_columnconfigure(column, weight=1)


    def set(self, row, column, value):
        widget = self._widgets[row][column]
        widget.configure(text=value)

    def set_width(self, column, width):
        self.column_widths[column] = width

    def add(self, values):
        current_row = []
        for col, v in enumerate(values):
            label = Label(self, text=v, borderwidth=0, width=self.column_widths[col])
            label.grid(row=self.rows, column=col, sticky="nsew", padx=self.padx, pady=self.pady)
            current_row.append(label)
        self._widgets.append(current_row)

