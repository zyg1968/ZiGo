import numpy as np
import random
import sys
import struct

N=19

if __name__ == '__main__':
    a=np.random.randint(0xff,0x20000000,(5,N*N))
    b=np.random.randint(0x41,0x43454532,(5,N*N))
    c=(a <<32)+b
    print(c)
    with open("zobrist.dt", "wb") as f:
        c.astype("int64").tofile(f)
                
    print("wc0")
