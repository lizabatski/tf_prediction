from multiprocessing import Pool
import os

def f(x):
    return (x, os.getpid())

if __name__ == "__main__":
    with Pool(4) as p:
        print(p.map(f, [1,2,3,4,5,6,7,8]))
