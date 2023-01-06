import numpy as np
import sys
import gc

def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

d = {"this dict": ["has a list", "of strings"]}

lst = [i for i in range(1000000)]
arr = np.array([lst])
pgs = lambda x, name="obj": print(f"{name} space: {actualsize(x) / 10**(3*2)} MB")

container = [arr for _ in range(1000)]
np_container = np.array(container)

pgs(lst, "list")
pgs(arr, "array")
pgs(container, "list of arrays")
pgs(np_container, "numpy list of arrays")

cont_size = actualsize(container)
per_arr = (cont_size - 72) / len(container)
print(f"Per array: {per_arr}")
# print(np_container)
