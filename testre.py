import re

SGF_POS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

s = '''
GM[1]
FF[4]
SZ[19]
PW[guojuan]
WR[5p]
PB[Bazill16]
BR[9k]
DT[2000-10-10]
PC[The Kiseido Go Server (KGS) at http://kgs.kiseido.com/]
KM[0.50]
RE[W+Time]
HA[9]RU[Japanese]CA[UTF-8]TM[480]OT[6x30 byo-yomi]AB[dd][jd][pd][dj][jj][pj][dp][jp][pp]
'''

def parse_move(coor):
    if coor=='' or coor == ' ' or coor == 'tt':
        return go.PASS
    return SGF_POS.index(coor[0]), SGF_POS.index(coor[1])


def parse_props(s):
    ms = re.findall(r'([A-Za-z]+)((\[([^\[\]]+)?\])+)',s)
    if not ms:
        return None
    props = {}
    for k,mv,_,v in ms:
        if re.match(r'(\[[^\[\]]+\]){2,}', mv):
            props[k] = mv
            ms1 = re.findall(r'\[([^\[\]]+?)\]', mv)
            ss = list(map(lambda x: parse_move(x), ms1))
            print(ss)
        else:
            props[k] = v
    return props

if __name__ == '__main__':
    p = parse_props(s)
    print(p)
