import multiprocessing
import time
import numpy as np

def fun(a:np.ndarray,b:np.ndarray):
    return a * b

tic = time.time()
pool = multiprocessing.Pool(processes=4)
iterable = ((np.full([1,3],1), np.full([1,3],2)),
            (np.full([1,3],3), np.full([1,3],4)),
            (np.full([1,3],5), np.full([1,3],6)))
result = pool.starmap_async(fun, iterable).get()
pool.close()
pool.join()
print(time.time() - tic)
print(result, sum(result))