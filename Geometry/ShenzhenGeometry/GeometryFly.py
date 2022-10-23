import os
import yaml
import h5py
import scipy.io as sco
import torch
import numpy as np
import torch.multiprocessing as multiprocessing
import JITShenzhenGeometry as projector
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
def threadFun(mode:str, target:torch.Tensor, projectVector:torch.Tensor, detectorSize:torch.Tensor, volumeSize:torch.Tensor, systemNum:int, device:int):
    if mode == "forward": result = projector.forward(target.to(device), volumeSize.to(device), detectorSize.to(device), projectVector.to(device), systemNum, device)
    elif mode == "backward": result = projector.backward(target.to(device), volumeSize.to(device), detectorSize.to(device), projectVector.to(device), systemNum, device)
    return result.to(0)

def threadStart(mode:str, target:torch.Tensor, projectVector:torch.Tensor, detectorSize:torch.Tensor, volumeSize:torch.Tensor, systemNum:int, processNum:int):
    multiprocessing.set_start_method("spawn")
    pool = multiprocessing.Pool(processes=processNum)
    args = ((mode, target, projectVector[id::processNum,:], detectorSize, volumeSize, int(systemNum/processNum), id) for id in range(processNum))
    result = pool.starmap_async(threadFun, args).get()
    pool.close()
    pool.terminate()
    pool.join()
    return sum(result)

class ShenzhenGeometry(Geometry):
    def __init__(self):
        self.anglesNum = params["anglesNum"]
        self.systemNum = params["fly"]
        parameters = sco.loadmat(os.path.join(params["root"],"proj_vec.mat"))
        # parameters = h5py.File(os.path.join(params["root"], "proj_vec.mat"), 'r')
        self.projectVector = np.array(parameters['proj_vec'])
        self.projectVector = self.projectVector[:2048:8,:]
        self.projectVector[:, [0, 1, 3, 4, 6, 7, 9, 10]] /= 3
        self.projectVector[:, [2, 5]] += 0
        self.projectVector[:, [2, 5]] /= 3
        self.projectVector = torch.from_numpy(self.projectVector).float().contiguous()
        detectorSize = torch.tensor(params["detectorSize"], dtype=torch.int32)
        volumeSize = torch.tensor(params["volumeSize"], dtype=torch.int32)
        super(ShenzhenGeometry, self).__init__(volumeSize, detectorSize)
        self.weight = torch.ones([1, 1, volumeSize[2], volumeSize[1], volumeSize[0]])
        self.weight = projector.forward(self.weight.to(0), self.volumeSize.to(0), self.detectorSize.to(0), self.projectVector[::self.systemNum,:].to(0), 1, 0)
        self.weight = projector.backward(self.weight, self.volumeSize.to(0), self.detectorSize.to(0), self.projectVector[::self.systemNum,:].to(0), 1, 0) + 1

    def fp(self, volume, device):
        sino = threadStart("forward", volume, self.projectVector, self.detectorSize, self.volumeSize, self.systemNum, 1)
        return sino

    def bp(self, sino, device):
        volume = threadStart("backward", sino, self.projectVector, self.detectorSize, self.volumeSize, self.systemNum, 1)
        return volume / self.weight.to(device)