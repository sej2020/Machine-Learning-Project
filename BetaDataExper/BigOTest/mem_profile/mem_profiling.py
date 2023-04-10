import numpy as np
import pandas as pd
import memray
from os import times


def np_alloc_stuff(elems, type, i):
    with memray.Tracker(f"testing_{i}.bin", native_traces=True):
        arr = np.ones((elems,), dtype=type)
        arr2 = arr + 3.5
    
    return arr

def pd_alloc_stuff(elems, type):
    with memray.Tracker(f"testing_{times[1]}.bin"):
        arr = np_alloc_stuff(elems, type)
        df = pd.Series(arr)
    
    return  df

if __name__=="__main__":
    e = 1000000
    t = np.int8
    for i in range(10):
        np_alloc_stuff(e, t, i)
    print("finito")