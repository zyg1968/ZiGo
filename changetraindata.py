import os
import sys
import gzip
import struct
import math
import numpy as np

def main():
    for fn in os.listdir("traindata"):
        ss = None
        mvs = None
        wrs = None
        data_size=0
        batch_size=0
        batch_num = 0
        header_bytes=None
        fp = os.path.join("traindata", fn)
        if not os.path.isfile(fp):
            continue
        print('读训练数据文件{}'.format(fp), file=sys.stderr)
        with gzip.open(fp, "rb") as f:
            header_bytes = f.read(struct.calcsize('ii'))
            data_size, batch_size = struct.unpack('ii', header_bytes)
            batch_num = math.ceil(data_size/batch_size)
            for i in range(batch_num):
                states = np.load(f) 
                move_values = np.load(f)
                move_values = np.concatenate((move_values[:,0:81],move_values[:,-2:]), axis=1)
                win_rates = np.load(f)
                ss = np.concatenate((ss, states), axis=0) if ss is not None else states
                mvs = np.concatenate((mvs, move_values), axis=0) if mvs is not None else move_values
                wrs = np.concatenate((wrs, win_rates), axis=0) if wrs is not None else win_rates
            assert len(f.read()) == 0
            f.close()
        print('保存训练数据到%s中……' % (fp), file=sys.stderr)
        fp = os.path.join("traindata/f2", fn)
        with gzip.open(fp, "wb", compresslevel=6) as f:
            f.write(header_bytes)
            for i in range(batch_num):
                s=i*batch_size
                e=min((i+1)*batch_size, data_size)
                np.save(f, ss[s:e])
                np.save(f, mvs[s:e])
                np.save(f, wrs[s:e])
            f.close()

    print("完成修改！")

if __name__ == "__main__":
    main()
