import torch
import numpy as np

def l1ball(v,b):
    v = v.cpu().numpy()
    if (np.sum(np.abs(v)) < b):
        return torch.from_numpy(v).cuda()
    paramLambda = 0
    objectValue = np.sum(np.max(np.abs(v)-paramLambda,0)) - b
    iterations = 0
    while(np.abs(objectValue) > 1e-4 and iterations < 100):
        objectValue = np.sum(np.max(np.abs(v) - paramLambda, 0)) - b
        difference = np.sum(-np.where(np.abs(v)-paramLambda > 0,1,0)) + 0.001
        paramLambda -= objectValue / difference
        iterations += 1
    paramLambda = np.max(paramLambda, 0)
    w = np.sign(v) * np.max(np.abs(v)-paramLambda,0)
    return torch.from_numpy(w).cuda()