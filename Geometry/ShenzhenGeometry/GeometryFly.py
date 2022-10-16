import os
import yaml
import h5py
import scipy.io as sco
import torch
import numpy as np
import JITShenzhenGeometry as projector
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class ShenzhenGeometry(Geometry):
    def __init__(self):
        self.anglesNum = params["anglesNum"]
        self.systemNum = params["fly"]
        parameters = sco.loadmat(os.path.join(params["root"],"proj_vec.mat"))
        self.projectVector = parameters['proj_vec']
        self.projectVector[:, [0, 1, 3, 4, 6, 7, 9, 10]] /= 3
        self.projectVector[:, [2, 5]] += 200
        self.projectVector[:, [2, 5]] /= 4
        self.projectVector = torch.from_numpy(self.projectVector).float().contiguous()
        detectorSize = torch.tensor(params["detectorSize"], dtype=torch.int32)
        volumeSize = torch.tensor(params["volumeSize"], dtype=torch.int32)
        super(ShenzhenGeometry, self).__init__(volumeSize, detectorSize)

    def fp(self, volume, device):
        sino = projector.forward(volume, self.volumeSize.to(device), self.detectorSize.to(device),
                                 self.projectVector.to(device), self.systemNum, int(device[-1]))
        return sino

    def bp(self, sino, device):
        volume = 0
        for H in self.Hs:
            volume += len(self.Hs) * H.T * H * np.ones(self.volumeSize).flatten()
        volume /= self.weight
        return torch.from_numpy(volume.reshape(self.torchVolumeSize)).to(device)