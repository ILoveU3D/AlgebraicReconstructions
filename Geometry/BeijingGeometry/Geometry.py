import os
import yaml
import astra
import torch
import numpy as np
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class BeijingGeometry(Geometry):
    def __init__(self):
        anglesNum = params["anglesNum"]
        files = os.listdir(params["anglesRoot"])
        files = sorted(files, key=lambda x:float(x.split('_')[2]))
        angles = np.array([float(item.split('_')[2]) for item in files]) * np.pi / 180
        log = open(os.path.join(params["anglesRoot"], "..", "files.txt"), 'w')
        SID = params["SID"]
        SOD = params["SOD"]
        pixelSpacing = params["pixelSpacing"]
        R1 = SOD / pixelSpacing
        R2 = (SID - SOD) / pixelSpacing
        offset = params["zOffset"]
        detectorSize = params["detectorSize"]
        volumeSize = params["volumeSize"]
        thetaDec = detectorSize[0] / (2 * (SID - SOD) / pixelSpacing)
        BG = (SID - SOD) / pixelSpacing * np.sin(thetaDec)
        AG = (SID - SOD) / pixelSpacing * np.cos(thetaDec) + SOD / pixelSpacing
        AB = np.sqrt(BG**2 + AG**2)
        radius = BG / AB * SOD / pixelSpacing * 2
        sampleInterval = radius / volumeSize[0]
        z0 = offset / pixelSpacing * SOD / SID
        z1 = (SOD / pixelSpacing - radius / 2) * (offset / pixelSpacing - detectorSize[1]/2)/(SID/pixelSpacing)
        z2 = (SOD / pixelSpacing + radius / 2) * (offset / pixelSpacing + detectorSize[1]/2)/(SID/pixelSpacing)
        zOffsetSrc = -offset / pixelSpacing * SOD / SID - (z1+z2)/2 + z0
        zOffsetDet = offset / pixelSpacing * (1 - SOD / SID) - (z1+z2)/2 + z0
        sliceInterval = (z2 - z1) / volumeSize[2]
        projectVector = np.zeros((anglesNum * detectorSize[0], 12))
        for i in range(0, len(angles), int(len(angles) / anglesNum)):
            for j in range(detectorSize[0]):
                corr = j * anglesNum + int(i / len(angles) * anglesNum)
                projectVector[corr, 0] = R1 * np.cos(angles[i] + np.pi) / sampleInterval
                projectVector[corr, 1] = R1 * np.sin(angles[i] + np.pi) / sampleInterval
                projectVector[corr, 2] = zOffsetSrc / sliceInterval
                projectVector[corr, 3] = R2 * np.cos(angles[i] + (j - detectorSize[0]/2 - 0.5)/R2) / sampleInterval
                projectVector[corr, 4] = R2 * np.sin(angles[i] + (j - detectorSize[0]/2 - 0.5)/R2) / sampleInterval
                projectVector[corr, 5] = zOffsetDet / sliceInterval
                projectVector[corr, 6] = 0
                projectVector[corr, 7] = 0
                projectVector[corr, 8] = 1 / sliceInterval
                projectVector[corr, 9] = np.cos(angles[i] - (j - detectorSize[0]/2 - 0.5)/R2 - np.pi/2) / sampleInterval
                projectVector[corr, 10] = np.sin(angles[i] - (j - detectorSize[0]/2 - 0.5)/R2 - np.pi/2) / sampleInterval
                projectVector[corr, 11] = 0
            log.write(files[i] + '\n')
        log.close()
        projectorGeometry = astra.create_proj_geom('cone_vec',1,detectorSize[1], projectVector)
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1],volumeSize[2])
        projector = astra.create_projector('cuda3d',projectorGeometry,volumeGeometry)
        detectorSize.append(anglesNum)
        super(BeijingGeometry, self).__init__(volumeSize, detectorSize, astra.OpTomo(projector))
        self.weight = self.H.T * self.H * np.ones(self.volumeSize).flatten() + 1
        self.torchVolumeSize = [1, volumeSize[2], volumeSize[1], volumeSize[0]]
        self.torchDetectorSize = [1, detectorSize[2], detectorSize[1], detectorSize[0]]

    def fp(self, volume, device):
        sino = self.H * volume.cpu().flatten()
        return torch.from_numpy(sino.reshape(self.torchDetectorSize)).to(device)

    def bp(self, sino, device):
        volume = self.H.T * sino.cpu().flatten() / self.weight
        return torch.from_numpy(volume.reshape(self.torchVolumeSize)).to(device)