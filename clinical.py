import torch
import numpy as np
from Geometry.ShenzhenGeometry.Geometry import ShenzhenGeometry
from Algorithm.DTV.CP import DTVCP

geometry = ShenzhenGeometry()
dtv = DTVCP(geometry)

data = np.fromfile("/media/seu/wyk/Data/raws/proj.raw", dtype="float32")
data = np.reshape(data, geometry.torchDetectorSize)
data = torch.from_numpy(data)
output = dtv.run(geometry.bp(data, dtv.device), data)