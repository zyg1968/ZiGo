import sqlite3
import numpy as np
import dbhash
import go
import config
import board

def convert_list(s):
    b = str.encode(s)
    l = list(b)
    return l

config.read_cfg("config")
go.set_board_size(config.board_size)

conn=sqlite3.connect("zigo-9.db", isolation_level=None)
cur = conn.cursor()
sqlstr = "SELECT b.points,a.stones FROM stones AS a,points AS b WHERE a.pid=b.pid AND a.hash IN (?,?,?,?,?,?,?,?)"
hashstr = [4177649083943021140,2,3,4,5,6,7,8]
idstr = [1,2,3,4,5,6,7,8]
cur.execute(sqlstr,hashstr)
rows = cur.fetchall()
#print(rows)
if rows:
    ss = convert_list(rows[0][1])
    print(ss)
    ss = np.array(ss)
    ss=ss.reshape((9,9))
    print(ss)
    stones = list(ss)
    hash = dbhash.calc_hash(stones, 1, None)
    print(hash)
    pos=board.Board()
    hash1 = dbhash.get_hash(pos)
    print(pos.stones)
    print(hash1)
print("ok")
cur.close()
conn.close()
