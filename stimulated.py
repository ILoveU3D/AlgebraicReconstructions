import torch
import numpy as np
# from Geometry.ShenzhenGeometry.GeometryStationary import ShenzhenGeometry
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
from Geometry.StandardGeometry.Geometry import StandardGeometry
from Geometry.BeijingGeometry.Geometry import BeijingGeometry
from Utils.AstraConvertion import astra2Projection
from Algorithm.DTV.CP import DTVCP

geometry = StandardGeometry()
dtv = DTVCP(geometry)

data = np.fromfile("/home/nv/wyk/raws/trainData/1.raw", dtype="float32")
data = np.reshape(data, geometry.torchVolumeSize)
data = torch.from_numpy(data)
sino = geometry.fp(data, dtv.device)
output = dtv.run(geometry.bp(sino, dtv.device), sino)