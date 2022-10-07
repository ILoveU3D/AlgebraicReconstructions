import os
import yaml
import astra
import torch
import numpy as np
from torch.autograd import Function
import JITStandardGeometry as projector
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class StandardGeometry(Geometry):
    def __init__(self):
        self.angles = np.linspace(0, 360, params["angles"], False).tolist()
        self.angles = torch.tensor(self.angles, dtype=torch.int32)
        detectorSize = torch.tensor(params["detectorSize"], dtype=torch.int32)
        volumeSize = torch.tensor(params["volumeSize"], dtype=torch.int32)
        self.SID = params["SID"]
        self.SDD = params["SDD"]
        super(StandardGeometry, self).__init__(volumeSize, detectorSize)

    def fp(self, volume, device):
        sino = projector.forward(volume, self.angles.to(device), self.volumeSize.to(device), self.detectorSize.to(device), self.SID, self.SDD, int(device[-1]))
        return sino

    def bp(self, sino, device):
        volume = projector.backward(sino, self.angles.to(device), self.volumeSize.to(device), self.detectorSize.to(device), self.SID, self.SDD, int(device[-1]))
        return volume