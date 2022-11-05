import numpy as np

def append_word(msg: str) -> np.ndarray:
    """
    This function does very administrative tasks
    """
    return np.zeros((2, 10))


if __name__=="__main__":
    message = "My name is "
    result = append_word(message)

    a = (1, 2, 3)
    b = a[0]
    c = a[1]
    d = a[2]

    b, c, d = a

    for i in range(len(a)):
        print(a[i])

    for val in a:
        print(val)

    for i, val in enumerate(a):
        print(f"{i}: {val}")