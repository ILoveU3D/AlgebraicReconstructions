import os

import scipy.io
import yaml
import astra
import torch
import h5py
import numpy as np
from torch.autograd import Function
import JITShenzhenGeometry as projector
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class ShenzhenGeometry(Geometry):
    def __init__(self):
        anglesNum = params["anglesNum"]
        # parameters = scipy.io.loadmat(os.path.join(params["root"], "para.mat"))
        parameters = h5py.File(os.path.join(params["root"], "Para.mat"), 'r')
        det = np.transpose(parameters['det'])
        source = np.transpose(parameters['source'])
        u = np.transpose(parameters['u'])
        v = np.transpose(parameters['v'])
        self.projectVector = np.zeros((anglesNum, 12), dtype=np.float32)
        for i in range(anglesNum):
            self.projectVector[i, 0:3] = source[:, i]
            self.projectVector[i, 3:6] = det[:, i]
            self.projectVector[i, 6:9] = u[:, i]
            self.projectVector[i, 9:12] = v[:, i]
        self.projectVector[:, [0, 1, 3, 4, 6, 7, 9, 10]] /= 2
        self.projectVector[:, [2, 5]] += 300
        self.projectVector[:, [2, 5]] /= 2
        self.projectVector = torch.from_numpy(self.projectVector)
        detectorSize = torch.tensor(params["detectorSize"], dtype=torch.int32)
        volumeSize = torch.tensor(params["volumeSize"], dtype=torch.int32)
        super(ShenzhenGeometry, self).__init__(volumeSize, detectorSize)

    def fp(self, volume, device):
        sino = projector.forward(volume, self.volumeSize.to(device), self.detectorSize.to(device), self.projectVector.to(device), 1, int(device[-1]))
        return sino

    def bp(self, sino, device):
        volume = projector.backward(sino, self.volumeSize.to(device), self.detectorSize.to(device), self.projectVector.to(device), 1, int(device[-1]))
        return volume