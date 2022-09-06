import torch
import numpy as np

def l1ball(v,b):
    if (torch.sum(torch.abs(v)) < b):
        return v
    paramLambda = 0
    objectValue = torch.sum(torch.nn.functional.relu(torch.abs(v)-paramLambda)) - b
    iterations = 0
    while(torch.abs(objectValue) > 1e-4 and iterations < 100):
        objectValue = torch.sum(torch.nn.functional.relu(torch.abs(v) - paramLambda)) - b
        difference = torch.sum(-torch.where(torch.abs(v)-paramLambda > 0,1,0)) + 0.001
        paramLambda -= (objectValue / difference).item()
        iterations += 1
    paramLambda = np.max(paramLambda, 0)
    w = torch.sign(v) * torch.nn.functional.relu(torch.abs(v)-paramLambda)
    return w