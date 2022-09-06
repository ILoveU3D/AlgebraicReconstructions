import torch
import numpy as np
from Geometry.ShenzhenGeometry.Geometry import ShenzhenGeometry
from Geometry.StandardGeometry.Geometry import StandardGeometry
from Algorithm.DTV.CP import DTVCP

geometry = ShenzhenGeometry()
dtv = DTVCP(geometry)

data = np.fromfile("/media/seu/wyk/Data/raws/sample.raw", dtype="float32")
data = np.reshape(data, geometry.torchVolumeSize)
data = torch.from_numpy(data)
sino = geometry.fp(data, dtv.device)
output = dtv.run(geometry.bp(sino, dtv.device), sino)