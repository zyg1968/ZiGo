"""A ScrolledText widget feels like a text widget but also has a
vertical scroll bar on its right.  (Later, options may be added to
add a horizontal bar as well, to make the bars disappear
automatically when not needed, to move them to the other side of the
window, etc.)

Configuration options are passed to the Text widget.
A Frame widget is inserted between the master and the text, to hold
the Scrollbar widget.
Most methods calls are inherited from the Text widget; Pack, Grid and
Place methods are redirected to the Frame widget however.
"""

__all__ = ['ScrollText']

from tkinter import Frame, Text, Scrollbar, Pack, Grid, Place
from tkinter.constants import RIGHT, LEFT, Y, BOTH, END
import re
import go

class ScrollText(Text):
    def __init__(self, qp, master=None, log=None, **kw):
        self.qp=qp
        self.log=log
        self.frame = Frame(master)
        self.vbar = Scrollbar(self.frame)
        self.vbar.pack(side=RIGHT, fill=Y)

        kw.update({'yscrollcommand': self.vbar.set})
        Text.__init__(self, self.frame, **kw)
        self.pack(side=LEFT, fill=BOTH, expand=True)
        self.vbar['command'] = self.yview
        self.values={}

        # Copy geometry methods of self.frame without overriding Text
        # methods -- hack!
        text_meths = vars(Text).keys()
        methods = vars(Pack).keys() | vars(Grid).keys() | vars(Place).keys()
        methods = methods.difference(text_meths)

        for m in methods:
            if m[0] != '_' and m != 'config' and m != 'configure':
                setattr(self, m, getattr(self.frame, m))

    def __str__(self):
        return str(self.frame)

    def write(self, s):
        self.get_values(s)
        if self.values and self.qp:
            self.qp.update_heatmap(self.values)
        self.insert(END, s)
        txt = self.get(0.0, END)
        if txt.count('\n') > 200:
            self.save_log(0, 50)
            self.delete(0.0, 50.0)
        self.see(END)

    def flush(self):
        self.see(END)

    def save_log(self, start_line=0.0, end_line=END):
        if self.log:
            txt = self.get(start_line, end_line)
            with open(self.log, "a") as f:
                f.write(txt)

    def get_values(self, s):
        filtstr= r"\s*([A-Z]\d{1,2})\s\->\s+(\d+)\s*\(V:\s*([\d\.]+)\%\)\s\(N:\s*[\d\.]+\%\)\sPV:\s(.+)\n"   #"(Playouts: \d+, Win: ([\d\.]+)\%, PV: ([A-Z]\d{1,2})[\s\n])"|
        m = re.match(filtstr, s)
        fs1=r"all visits count: (\d+)\n"
        m1 = re.match(fs1, s)
        visitscount = 0
        if m1:
            self.values = {}
            visitscount = int(m1.group(1))
        if m:
            move = go.get_coor_from_gtp(m.group(1))
            data = go.AnalyseData(move, visitscount, int(m.group(2)), float(m.group(3)), m.group(4))
            self.values[move] = data
        return self.values
