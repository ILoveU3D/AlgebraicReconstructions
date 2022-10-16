import os
import yaml
import astra
import torch
import numpy as np
import JITBeijingGeometry as projector
from torch.utils.cpp_extension import load
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
# projector = load(name="beijing", extra_include_paths=["include"],
#                  sources=[os.path.join(os.path.dirname(__file__), "plug/kernel/jit.cpp"),
#                         os.path.join(os.path.dirname(__file__),"plug/kernel/cone_project_cuda_kernel.cu")])
class BeijingGeometry(Geometry):
    def __init__(self):
        anglesNum = params["anglesNum"]
        self.angles = torch.from_numpy(np.linspace(0,360,anglesNum,False))
        detectorSize = torch.tensor(params["detectorSize"])
        volumeSize = torch.tensor(params["volumeSize"])
        super(BeijingGeometry, self).__init__(volumeSize, detectorSize)
        self.SID = params["SID"]
        self.SDD = params["SDD"]
        self.pixelSpacing = params["pixelSpacing"]
        self.offset = params["zOffset"]

    def fp(self, volume, device):
        sino = projector.forward(volume, self.angles.to(device), self.volumeSize.to(device), self.detectorSize.to(device), self.SID, self.SDD, self.offset, self.pixelSpacing, int(device[-1]),1,1)
        return sino

    def bp(self, sino, device):
        ray = torch.from_numpy(self.ray).to(device)
        volume = projector.backward(sino, ray, self.angleNum, 512, 512, 72, -255, -255, -36, 769, 72, -384, -36,
                                              708 / 1.06, 1143 / 1.06, -27.5576 - 36, -100 / 1.06, 1, int(device[-1]))[0]
        return volume.reshape(self.torchVolumeSize) / self.weight.to(device)