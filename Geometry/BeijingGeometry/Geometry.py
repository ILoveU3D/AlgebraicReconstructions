import os
import yaml
import torch
import h5py
import scipy.io as sco
import numpy as np
import JITBeijingGeometry as projector
from torch.utils.cpp_extension import load
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class BeijingGeometry(Geometry):
    def __init__(self):
        parameters = sco.loadmat(os.path.join(params["root"], "projVec.mat"))['projection_matrix']
        # parameters = h5py.File(os.path.join(params["root"], "projVec.mat"), 'r')['projection_matrix']
        self.projectVector = np.array(parameters,dtype=np.float32)
        self.projectVector = torch.from_numpy(self.projectVector).contiguous()
        detectorSize = torch.tensor(params["detectorSize"], dtype=torch.int32)
        volumeSize = torch.tensor(params["volumeSize"], dtype=torch.int32)
        super(BeijingGeometry, self).__init__(volumeSize, detectorSize)
        # self.weight = torch.ones([1, 1, volumeSize[2], volumeSize[1], volumeSize[0]])
        # self.weight = projector.forward(self.weight.to(0), self.volumeSize.to(0), self.detectorSize.to(0),
        #                                 self.projectVector.to(0), 1, 0)
        # self.weight = projector.backward(self.weight, self.volumeSize.to(0), self.detectorSize.to(0),
        #                                  self.projectVector.to(0), 1, 0) + 1

    def fp(self, volume, device):
        sino = projector.forward(volume, self.volumeSize.to(device), self.detectorSize.to(device), self.projectVector.to(device), int(device[-1]))
        return sino

    def bp(self, sino, device):
        volume = projector.backward(sino, self.volumeSize.to(device), self.detectorSize.to(device), self.projectVector.to(device), int(device[-1]))
        return volume