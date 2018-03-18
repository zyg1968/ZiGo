import re
import go
import board
import os

# map from numerical coordinates to letters used by SGF

SGF_POS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
KOMIS = {'chinese':7.5, 'janpanese':6.5, 'koran':5.5}

def isfloat(s):
    m = re.match(r'^[0-9]*\.*[0-9]*$', s)
    if m:
        return True
    return False

def parse_props(s):
    ms = re.findall(r'([A-Za-z]+)((\[([^\[\]]+)?\])+)',s)
    if not ms:
        return None
    props = {}
    for k,mv,_,v in ms:
        if re.match(r'(\[[^\[\]]+\]){2,}', mv):
            props[k] = mv
        else:
            props[k] = v
    return props


def parse_stones(s):
    s = re.sub(r'\s*', '', s)
    ms = re.findall(r'(?<=;)([BW])\[([^()]*?)\](|[^()]+?|\s*\([\s\S]+\))(?=;|$)', s)
    if not ms:
        return None
    moves = []
    first_color = None
    for c, m, subtree in ms:
        if not first_color:
            first_color = c
        moves.append(Stone(c,m,subtree))
    return moves

def parse_move(coor):
    if coor=='' or coor == ' ' or coor == 'tt':
        return go.PASS
    return SGF_POS.index(coor[0]), SGF_POS.index(coor[1])

def unparse_move(coor):
    if coor == go.PASS:
        return 'tt'
    return SGF_POS[coor[0]] + SGF_POS[coor[1]]

def parse_color(cstr):
    return go.BLACK if cstr=='B' else go.WHITE

def unparse_color(c):
    return 'B' if c==go.BLACK else 'W'

def parse_result(r):
    if not r:
        return 0
    result = 0
    if r=='0':
        return 0
    if r:
        rs = r.split('+')
        wc = 1 if rs[0].upper()=='B' else -1
        if rs[1].upper().startswith('R'): #中盘胜
            result = (go.N*go.N+1)*wc
        elif rs[1].upper().startswith('T'):  #对方超时（Time）等
            result = (go.N*go.N+2)*wc
        elif isfloat(rs[1]):
            result = float(rs[1])*wc
        else:
            result = 0
    return result

def unparse_result(r):
    s = 'B+' if r>0 else 'W+'
    ra = abs(r)
    if ra == go.N*go.N+1:
        s += 'R'
    elif ra == go.N*go.N+2:
        s += 'Time'
    elif ra == 0:
        s = '0'
    else:
        s += str(ra)
    return s

def get_sgf_board(fn):
    if not os.path.isfile(fn):
        return None
    sp = SgfParser(filename=fn)
    return sp.get_board()

def save_board(board, fn):
    sp = SgfParser(board=board)
    sp.save(fn)
    
class Stone():
    def __init__(self, color, move, relevant=None):
        self.color = parse_color(color)
        self.move = parse_move(move)
        self.relevant = relevant

    def to_playermove(self):
        return go.PlayerMove(self.move, self.color, None, None)

class SgfParser():
    def __init__(self, filename=None, board=None, content=None):
        self.header = None
        self.stones = None
        self.content = None
        if filename:
            self.content = self.load(filename)
        elif content:
            self.content = content
            self.parse()
        elif board:
            self.from_board(board)

    def __len__(self):
        return len(self.stones)

    def __getitem__(self, k):
        return self.stones[k]

    def __iter__(self):
        return iter(self.stones)

    def header_prop(self, k):
        if self.header and k in self.header:
            if self.header[k].isdigit():
                return int(self.header[k])
            elif isfloat(self.header[k]):
                return float(self.header[k])
            return self.header[k]
        return None

    def parse(self, s=None):
        if s:
            self.content = s
        if not self.content:
            return
        s = re.sub('[ \r\n\t]*', '', self.content)
        m = re.match(r'^\(;([\s\S]+?)(;[\s\S]+)\)$', s)
        hstr, mstr = m.group(1), m.group(2)
        self.header = parse_props(hstr)
        self.stones = parse_stones(mstr)

    def save(self, fn):
        s = '(;'
        for k, v in self.header.items():
            s += "{k}[{v}]".format(k=k, v=v)
        for i, stone in enumerate(self.stones):
            if i % 16 == 0:
                s += '\n'
            s += ";{color}[{coords}]".format(color=unparse_color(stone.color), coords=unparse_move(stone.move))
            s += stone.relevant or ''
        s += ')'
        f = open(fn, 'w')
        f.write(s)
        f.close()

    def load(self, fn):
        s = None
        if os.path.isfile(fn):
            f = open(fn, 'r')
            s = f.read()
            f.close()
            self.parse(s)
        return s

    def replay_sgf(self):
        pos = board.Board(komi=self.header_prop('KM'))
        pos.first_color = self.stones[0].color
        pos.to_move = pos.first_color
        pos.player1_name = str(self.header_prop('PB'))
        pos.player2_name = str(self.header_prop('PW'))
        pos.handicap = self.header_prop('HA')
        pos.result = parse_result(self.header_prop('RE'))
        if pos.result == 0:
            return []
        ab = self.header_prop("AB")
        aw = self.header_prop("AW")
        if ab:
            ms = re.findall(r'\[([^\[\]]+?)\]', ab)
            ss = list(map(lambda x: parse_move(x), ms))
            go.place_stones(pos.stones, go.BLACK, ss)
            pos.build_groups()
        if aw:
            ms = re.findall(r'\[([^\[\]]+?)\]', aw)
            ss = list(map(lambda x: parse_move(x), ms))
            go.place_stones(pos.stones, go.WHITE, ss)
            pos.build_groups()
        for stone in self.stones:
            if not pos.replay_move(stone.move, stone.color):
                break
            yield pos

    def get_board(self):
        if not self.header or not self.stones:
            return None
        pos = board.Board(komi=self.header_prop('KM'))
        pos.first_color = self.stones[0].color
        pos.to_move = pos.first_color
        pos.player1_name = str(self.header_prop('PB'))
        pos.player2_name = str(self.header_prop('PW'))
        pos.handicap = self.header_prop('HA')
        pos.result = parse_result(self.header_prop('RE'))
        ab = self.header_prop("AB")
        aw = self.header_prop("AW")
        if ab:
            ms = re.findall(r'\[([^\[\]]+?)\]', ab)
            ss = list(map(lambda x: parse_move(x), ms))
            go.place_stones(pos.stones, go.BLACK, ss)
            pos.build_groups()
        if aw:
            ms = re.findall(r'\[([^\[\]]+?)\]', aw)
            ss = list(map(lambda x: parse_move(x), ms))
            go.place_stones(pos.stones, go.WHITE, ss)
            pos.build_groups()

        for stone in self.stones:
            pos.replay_move(stone.move, stone.color)
        return pos
    
    def from_board(self, pos):
        self.header = {}
        self.header['FF'] = '4'
        self.header['GM'] = '1'
        self.header['CA'] = 'utf-8'
        self.header['RE'] = unparse_result(pos.result)
        self.header['KM'] = str(pos.komi)
        self.header['HA'] = pos.handicap
        self.header['PB'] = pos.player1_name
        self.header['PW'] = pos.player2_name
        self.header['AP'] = 'ZiGo SgfParser'
        self.header['SZ'] = str(go.N)
        self.stones = []
        for pm in pos.recent:
            coords=unparse_move(pm.move)
            c = unparse_color(pm.color)
            self.stones.append(Stone(c, coords))


class GameTree(object):
    def __init__(self, parent, parser=None):
        self.parent = parent
        self.parser = parser
        if parser:
            self.setup()
        self.nodes = []
        self.children = []

    def setup(self):
        self.parser.start_gametree = self.my_start_gametree
        self.parser.end_gametree = self.my_end_gametree
        self.parser.start_node = self.my_start_node

    def my_start_node(self):
        if len(self.nodes) > 0:
            previous = self.nodes[-1]
        elif self.parent.__class__ == GameTree:
            previous = self.parent.nodes[-1]
        else:
            previous = None
        node = Node(self, previous, self.parser)
        if len(self.nodes) == 0:
            node.first = True
            if self.parent.__class__ == GameTree:
                if len(previous.variations) > 0:
                    previous.variations[-1].next_variation = node
                    node.previous_variation = previous.variations[-1]
                previous.variations.append(node)
            else:
                if len(self.parent.children) > 1:
                    node.previous_variation = self.parent.children[-2].nodes[0]
                    self.parent.children[-2].nodes[0].next_variation = node

        self.nodes.append(node)

    def my_start_gametree(self):
        self.children.append(GameTree(self, self.parser))

    def my_end_gametree(self):
        self.parent.setup()

    def __iter__(self):
        return NodeIterator(self.nodes[0])

    @property
    def root(self):
        # @@@ technically for this to be root, self.parent must be a Collection
        return self.nodes[0]

    @property
    def rest(self):
        class _:
            def __iter__(_):
                return NodeIterator(self.nodes[0].next)
        if self.nodes[0].next:
            return _()
        else:
            return None

    def output(self, f):
        f.write("(")
        for node in self.nodes:
            node.output(f)
        for child in self.children:
            child.output(f)
        f.write(")")


class Node:
    def __init__(self, parent, previous, parser=None):
        self.parent = parent
        self.previous = previous
        self.parser = parser
        if parser:
            self.setup()
        self.properties = {}
        self.next = None
        self.previous_variation = None
        self.next_variation = None
        self.first = False
        self.variations = []
        if previous and not previous.next:
            previous.next = self

    def setup(self):
        self.parser.start_property = self.my_start_property
        self.parser.add_prop_value = self.my_add_prop_value
        self.parser.end_property = self.my_end_property
        self.parser.end_node = self.my_end_node

    def my_start_property(self, identifier):
        # @@@ check for duplicates
        self.current_property = identifier
        self.current_prop_value = []

    def my_add_prop_value(self, value):
        self.current_prop_value.append(value)

    def my_end_property(self):
        self.properties[self.current_property] = self.current_prop_value

    def my_end_node(self):
        self.parent.setup()

    def output(self, f):
        f.write(";")
        for key, values in sorted(self.properties.items()):
            f.write(key)
            for value in values:
                if "\\" in value:
                    value = "\\\\".join(value.split("\\"))
                if "]" in value:
                    value = "\\]".join(value.split("]"))
                f.write("[%s]" % value)


