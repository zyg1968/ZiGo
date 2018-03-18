import re

filtstr= r"\s*([A-Z]\d{1,2})\s\->\s+(\d+)\s*\(V: ([\d\.]+)\%\)\s\(N:\s*[\d\.]+\%\)\sPV:\s(.+)\n"
s='''G18 ->      56 (V: 51.03%) (N:  7.72%) PV: G18 E14 O15 D15 G13 C12 D12
 G13 ->      14 (V: 49.12%) (N:  4.32%) PV: G13 G18 C3 D3 C4 C5 B5 B6
 O15 ->      13 (V: 49.21%) (N:  3.82%) PV: O15 G18 C3 D3 C4 C5 B5 B6
 P15 ->      19 (V: 51.10%) (N:  2.52%) PV: P15 N17 N16 M17 O18 M16 N15 M15 G13 G18 C3 D3
'''
m = re.match(filtstr, s)
if m:
    move = m.group(1)
    visits = int(m.group(2))
    winrate = float(m.group(3))
    nextmoves = m.group(4)
    print(move, visits, winrate, nextmoves)
