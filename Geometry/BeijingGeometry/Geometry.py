import os
import yaml
import astra
import torch
import numpy as np
import ConeProjectZ_cuda as projector
from torch.utils.cpp_extension import load
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
# projector = load(name="beijing", extra_include_paths=["include"],
#                  sources=[os.path.join(os.path.dirname(__file__), "plug/kernel/cone_project_cuda.cpp"),
#                         os.path.join(os.path.dirname(__file__),"plug/kernel/cone_project_cuda_kernel.cu")])
class BeijingGeometry(Geometry):
    def __init__(self):
        anglesNum = params["anglesNum"]
        files = os.listdir(params["anglesRoot"])
        files = sorted(files, key=lambda x:float(x.split('_')[2]))
        angles = np.array([float(item.split('_')[2]) for item in files]) * np.pi / 180 - np.pi / 2
        log = open(os.path.join(params["anglesRoot"], "..", "files.txt"), 'w')
        detectorSize = params["detectorSize"]
        volumeSize = params["volumeSize"]
        super(BeijingGeometry, self).__init__(volumeSize, detectorSize, None)
        self.torchVolumeSize = [1, volumeSize[2], volumeSize[1], volumeSize[0]]
        self.torchDetectorSize = [1, anglesNum, detectorSize[1], detectorSize[0]]
        self.angleNum = anglesNum
        self.ray = np.zeros(anglesNum*2, dtype=np.float32)
        step = len(angles) / anglesNum
        coor = 0
        s = 0
        while s < anglesNum:
            self.ray[s * 2] = np.cos(angles[int(coor)])
            self.ray[s * 2 + 1] = np.sin(angles[int(coor)])
            log.write(files[int(coor)] + '\n')
            s += 1
            coor += step
        log.close()
        ray = torch.from_numpy(self.ray).to(0)
        self.weight = projector.forward(torch.ones(self.torchVolumeSize).cuda(), ray, self.angleNum, 512, 512, 72, -255, -255, -36, 769, 72, -384, -36,
                                 708 / 1.06, 1143 / 1.06, -27.5576 - 36, -100 / 1.06, 1, 0)[0]
        self.weight = projector.backward(self.weight, ray, self.angleNum, 512, 512, 72, -255, -255, -36, 769, 72, -384, -36,
                                    708 / 1.06, 1143 / 1.06, -27.5576 - 36, -100 / 1.06, 1, 0)[0]

    def fp(self, volume, device):
        ray = torch.from_numpy(self.ray).to(device)
        sino = projector.forward(volume, ray, self.angleNum, 512, 512, 72, -255, -255, -36, 769, 72, -384, -36,
                                              708 / 1.06, 1143 / 1.06, -27.5576 - 36, -100 / 1.06, 1, int(device[-1]))[0]
        return sino.reshape(self.torchDetectorSize)

    def bp(self, sino, device):
        ray = torch.from_numpy(self.ray).to(device)
        volume = projector.backward(sino, ray, self.angleNum, 512, 512, 72, -255, -255, -36, 769, 72, -384, -36,
                                              708 / 1.06, 1143 / 1.06, -27.5576 - 36, -100 / 1.06, 1, int(device[-1]))[0]
        return volume.reshape(self.torchVolumeSize) / self.weight.to(device)