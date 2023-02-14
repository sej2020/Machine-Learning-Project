import multiprocessing as mp
import hashlib
from time import sleep
from random import random

def load_words(path):
    with open(path, encoding='utf-8') as file:
        return file.readlines()

def hash_word(word):
    hash_object = hashlib.sha512()
    byte_data = word.encode('utf-8')
    hash_object.update(byte_data)
    return hash_object.hexdigest()

def main():
    path = "C:/Users/18123/AppData/Roaming/Microsoft/Spelling/1m_words.txt"
    words = load_words(path)
    print(f'Loaded {len(words)} words from {path}')
    with mp.Pool(8) as pool:
        known_words = set(pool.map(hash_word, words))
    print(f'Done, with {len(known_words)} hashes')

def task(value):
    random_value = random()
    sleep(random_value)
    # print(f'task got {(value, random_value)}')
    return (value, random_value)

def task1(identifier):
    value = random()
    sleep(value)
    print(f'{identifier} generated {value}', flush=True)
    return (identifier, value)

def result_callback(result_iterator):
    for i, v in result_iterator:
        if v > 0.5:
            _ = pool.apply_async(task2, args=(i, v))

def task2(identifier, result):
    value = random()
    sleep(value)
    print(f'{identifier} with {result} generated {value}', flush=True)
    return (identifier, result, value)

if __name__ == '__main__':
    # main()
    # Iterate multiple results **********************************************
    # with mp.Pool() as pool:
    #     for result in pool.map(task, range(10)):
    #         print(f'got {result}')
    # One asynchronous function **********************************************
    # with mp.Pool() as pool:
    #     _ = pool.apply_async(task, args=(1,))
    #     pool.close()
    #     pool.join()
    # Multiple asynchronous tasks **********************************************
    # with mp.Pool() as pool:
    #     _ = pool.map_async(task, range(10))
    #     pool.close()
    #     pool.join()
    # Unordered results **********************************************
    # with mp.Pool() as pool:
    #     for result in pool.imap_unordered(task, range(10)):
    #         print(f'got {result}')
    # Only need result of fastest process **********************************************
    # with mp.Pool() as pool:
    #     it = pool.imap_unordered(task, range(10))
    #     result = next(it)
    #     print(f'got {result}')
    #*********************************************************************
    with mp.Pool() as pool:
        result = pool.map_async(task1, range(10), callback=result_callback)
        result.wait()
        pool.close()
        pool.join()
    print("All Done. ")