import os
import yaml
import astra
import h5py
import torch
import numpy as np
import torch.multiprocessing as multiprocessing
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)

def threadFun(mode:str, target:np.ndarray, projectVector:np.ndarray, detectorSize:list, volumeSize:list):
    projectVector[:, [0, 1, 3, 4, 6, 7, 9, 10]] /= 3
    projectVector[:, [2, 5]] += 200
    projectVector[:, [2, 5]] /= 4
    volumeGeometry = astra.create_vol_geom(volumeSize[0], volumeSize[1], volumeSize[2])
    projectorGeometry = astra.create_proj_geom('cone_vec', detectorSize[0], detectorSize[1], projectVector)
    projector = astra.create_projector('cuda3d', projectorGeometry, volumeGeometry)
    H = astra.OpTomo(projector)
    if mode == "forward":
        return H * target.flatten()
    elif mode == "backward":
        return H.T * target.flatten()
    elif mode == "ortho":
        return H.T * H * target.flatten()

def threadStart(mode:str, target:np.ndarray, projectVector:np.ndarray, detectorSize:list, volumeSize:list, systemNum:int, processNum:int):
    pool = multiprocessing.Pool(processes=processNum)
    args = ((mode, target, projectVector[:,id::systemNum].T, detectorSize, volumeSize) for id in range(systemNum))
    result = pool.starmap_async(threadFun, args).get()
    pool.close()
    pool.join()
    return sum(result)

class ShenzhenGeometry(Geometry):
    def __init__(self):
        anglesNum = params["anglesNum"]
        parameters = h5py.File(os.path.join(params["root"],"proj_vec.mat"), 'r')
        detectorSize = params["detectorSize"]
        volumeSize = params["volumeSize"]
        detectorSize.append(anglesNum)
        super(ShenzhenGeometry, self).__init__(volumeSize, detectorSize)
        multiprocessing.set_start_method("spawn")
        self.projVec = parameters['proj_vec']
        self.systemNum = params["fly"]
        self.processNum = params["processNum"]
        self.weight = 1
        self.weight += threadStart("ortho", np.ones(self.volumeSize), self.projVec, self.detectorSize, self.volumeSize, self.systemNum, self.processNum)
        self.torchVolumeSize = [1, volumeSize[2], volumeSize[1], volumeSize[0]]
        self.torchDetectorSize = [1, detectorSize[2], detectorSize[1], detectorSize[0]]

    def fp(self, volume, device):
        sino = threadStart("forward", volume.cpu(), self.projVec, self.detectorSize, self.volumeSize, self.systemNum, self.processNum)
        return torch.from_numpy(sino.reshape(self.torchDetectorSize)).to(device)

    def bp(self, sino, device):
        volume = threadStart("backward", sino.cpu(), self.projVec, self.detectorSize, self.volumeSize, self.systemNum, self.processNum)
        volume /= self.weight
        return torch.from_numpy(volume.reshape(self.torchVolumeSize)).to(device)