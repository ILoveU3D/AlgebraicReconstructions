import os
import yaml
import astra
import h5py
import torch
import numpy as np
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class ShenzhenGeometry(Geometry):
    def __init__(self):
        anglesNum = params["anglesNum"]
        systemNum = params["fly"]
        parameters = h5py.File(os.path.join(params["root"],"proj_vec.mat"), 'r')
        projVec = parameters['proj_vec']
        # parameters = h5py.File(os.path.join(params["root"], "Para.mat"), 'r')
        # det = np.transpose(parameters['det'])
        # source = np.transpose(parameters['source'])
        # u = np.transpose(parameters['u'])
        # v = np.transpose(parameters['v'])
        # projVec = np.concatenate((source, det, u, v))[:,:systemNum*anglesNum*2:2]
        self.H= None
        detectorSize = params["detectorSize"]
        volumeSize = params["volumeSize"]
        volumeGeometry = astra.create_vol_geom(volumeSize[0], volumeSize[1], volumeSize[2])
        for id in range(systemNum):
            projectVector = projVec[:,id::systemNum].T
            projectVector[:, [0, 1, 3, 4, 6, 7, 9, 10]] /= 3
            projectVector[:, [2, 5]] += 200
            projectVector[:, [2, 5]] /= 4
            projectorGeometry = astra.create_proj_geom('cone_vec', detectorSize[0],detectorSize[1], projectVector)
            projector = astra.create_projector('cuda3d',projectorGeometry,volumeGeometry)
            if self.H is None: self.H = astra.OpTomo(projector)
            else: self.H = self.H + astra.OpTomo(projector)
        detectorSize.append(anglesNum)
        super(ShenzhenGeometry, self).__init__(volumeSize, detectorSize, self.H)
        self.weight = 1
        self.weight += 1 / systemNum * self.H.T * self.H * np.ones(self.volumeSize).flatten()
        self.systemNum = systemNum
        # for H in self.Hs:
        #     self.weight += 1 / systemNum * H.T * H * np.ones(self.volumeSize).flatten()
        self.torchVolumeSize = [1, volumeSize[2], volumeSize[1], volumeSize[0]]
        self.torchDetectorSize = [1, detectorSize[2], detectorSize[1], detectorSize[0]]

    def fp(self, volume, device):
        sino = 1 / self.systemNum * self.H * volume.cpu().flatten()
        return torch.from_numpy(sino.reshape(self.torchDetectorSize)).to(device)

    def bp(self, sino, device):
        volume = 1 / self.systemNum * self.H.T * sino.cpu().flatten()
        volume /= self.weight
        return torch.from_numpy(volume.reshape(self.torchVolumeSize)).to(device)