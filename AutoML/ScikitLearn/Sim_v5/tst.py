from time import process_time as pt
from random import randint


def f():
    rand = randint(0, 10)
    return rand

def looper():
    for i in range(10000):
        start = pt()
        var = i ** 2
        stop = pt()
        time_dict[f"run_{i}"] = stop - start
    
    return

def main():
    global time_dict
    time_dict = {}
    start = pt()
    i = f()
    stop = pt()
    time_dict["f_time"] = stop - start
    looper()

if __name__ == "__main__":
    main()
    print(time_dict)