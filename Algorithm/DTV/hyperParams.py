import numpy as np

"""
Power method to find L2 norm
"""
def power_method(K):
    N = K.shape[1]
    x = np.random.rand(N)
    for i in range(10):
        x = K.T * K * x
        x = x / np.linalg.norm(x, 2)
        s = np.linalg.norm(K * x, 2)
    return s

def power_method_K(matrixs):
    N = matrixs[0].shape[1]
    x = np.random.rand(N)
    for i in range(10):
        x = sum([M.T * M * x for M in matrixs])
        x = x / np.linalg.norm(x, 2)
        s = np.sqrt(np.sum([pow(np.linalg.norm(M * x, 2), 2) for M in matrixs]))
    return s