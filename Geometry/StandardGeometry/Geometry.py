import os
import yaml
import astra
import numpy as np
from ..Geometry import Geometry

params = yaml.load(open(os.path.join(os.path.dirname(__file__), r"params.yaml")), yaml.FullLoader)
class StandardGeometry(Geometry):
    def __init__(self):
        angles = np.linspace(0, 2 * np.pi, params["angles"], False)
        detectorSize = params["detectorSize"]
        projectorGeometry = astra.create_proj_geom('cone', 1, 1, detectorSize[0],detectorSize[1], angles, params["SID"], params["SOD"])
        volumeSize = params["volumeSize"]
        volumeGeometry = astra.create_vol_geom(volumeSize[0],volumeSize[1],volumeSize[2])
        projector = astra.create_projector('cuda3d',projectorGeometry,volumeGeometry)
        detectorSize.append(params["angles"])
        super(StandardGeometry, self).__init__(volumeSize, detectorSize, astra.OpTomo(projector))
        self.weight = self.H.T * self.H * np.ones(self.volumeSize).flatten()

    def fp(self, volume):
        sino = self.H * volume.flatten()
        return sino.reshape(self.detectorSize)

    def bp(self, sino):
        volume = self.H.T * sino.flatten() / self.weight
        return volume.reshape(self.volumeSize)