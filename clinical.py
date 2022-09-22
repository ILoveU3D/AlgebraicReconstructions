import torch
import numpy as np
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
from Geometry.BeijingGeometry.Geometry import BeijingGeometry
from Algorithm.DTV.CP import DTVCP
from Utils.AstraConvertion import projection2Astra

geometry = BeijingGeometry()
# dtv = DTVCP(geometry)

data = np.fromfile("/media/seu/wyk/Data/raws/proj.raw", dtype="float32")
data = np.reshape(data, geometry.detectorSize)
data = torch.from_numpy(data)
data = torch.reshape(projection2Astra(data), geometry.torchDetectorSize)
output = geometry.bp(data, "cuda:0")
output.detach().cpu().numpy().tofile("/media/seu/wyk/Data/raws/r.raw")
# output = dtv.run(geometry.bp(data, dtv.device), data)