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
    sino = projector.forward(target.to(device), volumeSize.to(device), detectorSize.to(device), projectVector.to(device), systemNum, device)
    return sino.to(0)

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
        # parameters = sco.loadmat(os.path.join(params["root"],"proj_vec.mat"))
        parameters = h5py.File(os.path.join(params["root"], "proj_vec.mat"), 'r')
        self.projectVector = np.array(parameters['proj_vec']).T
        self.projectVector = self.projectVector[:2048,:]
        self.projectVector[:, [0, 1, 3, 4, 6, 7, 9, 10]] /= 3
        self.projectVector[:, [2, 5]] += 200
        self.projectVector[:, [2, 5]] /= 4
        self.projectVector = torch.from_numpy(self.projectVector).float().contiguous()
        detectorSize = torch.tensor(params["detectorSize"], dtype=torch.int32)
        volumeSize = torch.tensor(params["volumeSize"], dtype=torch.int32)
        super(ShenzhenGeometry, self).__init__(volumeSize, detectorSize)

    def fp(self, volume, device):
        # sino = projector.forward(volume, self.volumeSize.to(device), self.detectorSize.to(device),
        #                          self.projectVector.to(device), self.systemNum, int(device[-1]))
        sino = threadStart("forward", volume, self.projectVector, self.detectorSize, self.volumeSize, self.systemNum, 4)
        return sino

    def bp(self, sino, device):
        volume = projector.backward(sino, self.volumeSize.to(device), self.detectorSize.to(device),
                                 self.projectVector.to(device), self.systemNum, int(device[-1]))
        return volume