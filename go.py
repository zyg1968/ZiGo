'''
A board is a NxN numpy array.
A Coordinate is a tuple index into the board.
A Move is a (Coordinate c | None).
A PlayerMove is a (Color, Move, Captured, Score) object

(0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.
'''
from collections import namedtuple
import copy
import itertools
import gtp
import numpy as np

MAX_BOARD = 19

EMPTY, BLACK, WHITE, BORDER, KO, FILL, UNKNOWN = range(0, 7)

MISSING_GROUP_ID = -1

class IllegalMove(Exception): pass

# these are initialized by set_board_size
N = None
ALL_COORDS = []
EMPTY_BOARD = None
NEIGHBORS = {}
DIAGONALS = {}
COLLUM_STR = ''
PASS = None
RESIGN = None


def set_board_size(n):
    '''
    Hopefully nobody tries to run both 9x9 and 19x19 game instances at once.
    Also, never do "from go import N, W, ALL_COORDS, EMPTY_BOARD".
    '''
    global N, ALL_COORDS, EMPTY_BOARD, NEIGHBORS, DIAGONALS, COLLUM_STR, PASS, RESIGN
    if N == n: return
    N = n
    PASS = (N, 0)
    RESIGN = (N, 1)
    ALL_COORDS = [(i, j) for i in range(n) for j in range(n)]
    EMPTY_BOARD = np.zeros([n, n], dtype=np.int8)
    def check_bounds(c):
        return c[0] % n == c[0] and c[1] % n == c[1]

    NEIGHBORS = {(x, y): list(filter(check_bounds, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_COORDS}
    DIAGONALS = {(x, y): list(filter(check_bounds, [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])) for x, y in ALL_COORDS}

    COLLUM_STR='ABCDEFGHJKLMNOPQRSTUVWXYZ'

class AnalyseData(namedtuple('AnalyseData', ['move', 'allvisits', 'visits', 'winrate','nextmoves'])): pass

def result_str(result):
    ar = abs(result)
    s = '中盘胜'
    if ar == N*N+1:
        s = get_color_str(result) + '中盘胜' 
    elif ar == N*N+2:
        s = get_color_str(-result) + '方超时负' 
    elif ar == 0:
        s = '和棋'
    else:
        s = '%s胜%.1f子' % (get_color_str(result), ar)
    return s

def get_point_str(p):
    return "{}领先{:.1f}目".format(get_color_str(p), abs(p))

def get_points_str(p, c):
    return get_point_str(p if c==BLACK else -p)

def get_color_str(color):
    if color not in [BLACK,WHITE]:
        return '黑' if color>1 else '白'
    return '黑' if color==BLACK else '白'

def get_legal_str(ind):
    ills = ['合法', '有子', '打劫(全局同形)', '自杀', '填眼', '不该PASS', '该PASS']
    return ills[ind]

def oppo_color(color):
    return BLACK+WHITE - color

def get_coor_from_gtp(cmd):
    if cmd.lower()=="pass":
        return PASS
    if cmd.lower()=="resign":
        return RESIGN
    if len(cmd)<2 or len(cmd)>3 or cmd[0] not in COLLUM_STR:
        return None
    y = COLLUM_STR.index(cmd[0])
    x = N - int(cmd[1:])
    return (x,y)

def flatten_coords(c):
    if not c:
        c=PASS
    r = N * c[0] + c[1]
    if r>N*N+1:
        return N*N
    return r

def unflatten_coords(f):
    if f<0 or f>N*N+2:
        return None
    return divmod(f, N)

def take_n(n, iterable):
    return list(itertools.islice(iterable, n))

def get_coor_from_vertex(vertex):
    if vertex == gtp.PASS:
        return PASS
    elif vertex == gtp.RESIGN:
        return RESIGN
    return (N - vertex[1], vertex[0]-1)

def get_vertex_from_coor(coor):
    if coor is None or coor==PASS:
        return gtp.PASS
    elif coor==RESIGN:
        return gtp.RESIGN
    return (coor[1]+1, N-coor[0])

def get_cmd_from_coor(coor):
    if coor[0] in range(N) and coor[1] in range(N):
        return COLLUM_STR[coor[1]]+str(N-coor[0])
    elif coor==RESIGN:
        return 'RESIGN'
    return 'PASS'

def get_cmd_from_vertex(vertex):
    if vertex[0] in range(1, N+1) and vertex[1] in range(1, N+1):
        return COLLUM_STR[vertex[0]-1]+str(vertex[1])
    elif vertex == gtp.RESIGN:
        return 'RESIGN'
    return 'PASS'

set_board_size(19)
