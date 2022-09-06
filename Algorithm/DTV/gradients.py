import torch
import numpy as np
from scipy.sparse import csc_matrix
from .hyperParams import power_method,power_method_K

matrixSet = []

def spraseMatrixX(volumeSize:tuple):
    N = volumeSize[0] * volumeSize[1] * volumeSize[2]
    row,col,data = _init(N)
    for i in range(N):
        if ((i + 1) % volumeSize[0] != 0):
            row.append(i)
            col.append(i + 1)
            data.append(1.0)
    return __getSpraseMatrix(row,col,data,N)

def spraseMatrixY(volumeSize:tuple):
    N = volumeSize[0] * volumeSize[1] * volumeSize[2]
    row, col, data = _init(N)
    for i in range(N - volumeSize[1]):
        row.append(i)
        col.append(i + volumeSize[1])
        data.append(1.0)
    return __getSpraseMatrix(row,col,data,N)

def spraseMatrixZ(volumeSize:tuple):
    N = volumeSize[0] * volumeSize[1] * volumeSize[2]
    row, col, data = _init(N)
    for i in range(N - volumeSize[1] * volumeSize[0]):
        row.append(i)
        col.append(i + volumeSize[1] * volumeSize[0])
        data.append(1.0)
    return __getSpraseMatrix(row,col,data,N)

def spraseMatrixI(volumeSize:tuple):
    N = volumeSize[0] * volumeSize[1] * volumeSize[2]
    row, col, data = _init(N)
    return __getSpraseMatrix(row,col,data,N)

def _init(N:int):
    row = []
    col = []
    data = []
    for i in range(N):
        row.append(i)
        col.append(i)
        data.append(-1.0)
    return row,col,data

def __getSpraseMatrix(row,col,data,N):
    i = torch.LongTensor([row, col])
    v = torch.FloatTensor(data)
    D = torch.sparse.FloatTensor(i, v, (N,N)).cuda()
    i = torch.LongTensor([col, row])
    v = torch.FloatTensor(data)
    DT = torch.sparse.FloatTensor(i, v, (N,N)).cuda()
    matrix = csc_matrix((np.array(data), (np.array(row), np.array(col))), shape=(N, N))
    norm = power_method(matrix)
    matrixSet.append(matrix)
    return D,DT,norm

def getNormK(H, opts):
    matrixSet.insert(0, H)
    inputSet = [matrix * opts[idx] for idx,matrix in enumerate(matrixSet)]
    L = power_method_K(inputSet)
    matrixSet.clear()
    return L