import torch
import numpy as np
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
from Algorithm.DTV.CP import DTVCP

geometry = ShenzhenGeometry()
dtv = DTVCP(geometry)

data = np.fromfile("/home/nv/wyk/raws/proj3.raw", dtype="float32")
data = np.reshape(data, geometry.torchDetectorSize)
data = torch.from_numpy(data)
output = dtv.run(geometry.bp(data, dtv.device), data)