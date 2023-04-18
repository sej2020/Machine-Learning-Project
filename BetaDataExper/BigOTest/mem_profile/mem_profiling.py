import numpy as np
import pandas as pd
import memray
from os import times
from pathlib import Path
import sys

# def np_alloc_stuff(elems, type, i):
#     with memray.Tracker(f"testing_{i}.bin", native_traces=True):
#         arr = np.ones((elems,), dtype=type)
#         arr2 = arr + 3.5
    
#     return arr

# def pd_alloc_stuff(elems, type):
#     with memray.Tracker(f"testing_{times[1]}.bin"):
#         arr = np_alloc_stuff(elems, type)
#         df = pd.Series(arr)
    
#     return  df

def testing(output_path, shape):
    with memray.Tracker(output_path, native_traces=True):
        d = {f"col_{i}": [i*j for j in range(shape[1])] for i in range(shape[0])}
        df = pd.DataFrame(d)
        df += 1

if __name__=="__main__":
    rows = 10000
    cols = 1000
    print(f"Expected size for the allocation is {rows * cols * sys.getsizeof(int()) / 1000000} MB")
    # step = 1000
    # make_path = lambda rows, cols: Path.cwd() / f"mem_{rows*cols + cols}.bin"
    # for r in range(step, rows+step, step):
    #     for c in range(step, cols+step, step):
    #         path = make_path(r, c)
    #         t = testing(path, (r, c))
    t = testing(Path("mem.bin"), (rows, cols))
            
    print("all done, foo")