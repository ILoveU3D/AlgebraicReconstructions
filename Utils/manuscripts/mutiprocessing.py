import multiprocessing
import time
import numpy as np
from Geometry.ShenzhenGeometry.GeometryStationary import ShenzhenGeometry


def fun(p:np.ndarray):
    geometry = ShenzhenGeometry()
    return geometry.H * p.flatten()

pool = multiprocessing.Pool(processes=4)
iterable = (
    (np.ones([32,32,4]),),
    (np.ones([32,32,4]),),
    (np.ones([32,32,4]),),
)
result = pool.starmap_async(fun, iterable).get()
pool.close()
pool.join()

sum(result).tofile("/media/seu/wyk/Data/raws/r.raw")