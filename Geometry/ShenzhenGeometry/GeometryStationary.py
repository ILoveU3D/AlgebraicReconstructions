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
        parameters = h5py.File(os.path.join(params["root"],"Para.mat"), 'r')
        det = np.transpose(parameters['det'])
        source = np.transpose(parameters['source'])
        u = np.transpose(parameters['u'])
        v = np.transpose(parameters['v'])
        projectVector = np.zeros((anglesNum, 12))
        for i in range(anglesNum):
            projectVector[i, 0:3] = source[:,i]
            projectVector[i, 3:6] = det[:, i]
            projectVector[i, 6:9] = u[:, i]
            projectVector[i, 9:12] = v[:, i]
        projectVector[:,[0,1,3,4,6,7,9,10]] /= 2
        # projectVector[:,[2,5]] += 500
        projectVector[:,[2,5]] /= 2
        detectorSize = params["detectorSize"]
        projectorGeometry = astra.create_proj_geom('cone_vec', detectorSize[0],detectorSize[1], projectVector)
        volumeSize = params["volumeSize"]
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1],volumeSize[2])
        projector = astra.create_projector('cuda3d',projectorGeometry,volumeGeometry)
        detectorSize.append(anglesNum)
        super(ShenzhenGeometry, self).__init__(volumeSize, detectorSize, astra.OpTomo(projector))
        self.weight = self.H.T * self.H * np.ones(self.volumeSize).flatten() + 1
        self.torchVolumeSize = [1, volumeSize[2], volumeSize[1], volumeSize[0]]
        self.torchDetectorSize = [1, detectorSize[2], detectorSize[1], detectorSize[0]]

    def fp(self, volume, device):
        sino = self.H * volume.cpu().flatten()
        return torch.from_numpy(sino.reshape(self.torchDetectorSize)).to(device)

    def bp(self, sino, device):
        volume = self.H.T * sino.cpu().flatten() / self.weight
        return torch.from_numpy(volume.reshape(self.torchVolumeSize)).to(device)