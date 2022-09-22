import torch
import numpy as np
# from Geometry.ShenzhenGeometry.GeometryStationary import ShenzhenGeometry
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
from Geometry.StandardGeometry.Geometry import StandardGeometry
from Geometry.BeijingGeometry.Geometry import BeijingGeometry
from Utils.AstraConvertion import astra2Projection
from Algorithm.DTV.CP import DTVCP

geometry = BeijingGeometry()
dtv = DTVCP(geometry)

data = np.fromfile("/media/seu/wyk/Data/raws/sample.raw", dtype="float32")
data = np.reshape(data, geometry.torchVolumeSize)
data = torch.from_numpy(data)
sino = geometry.fp(data, dtv.device)
# astra2Projection(sino.reshape(geometry.detectorSize)).cpu().numpy().tofile("/media/seu/wyk/Data/raws/r.raw")
# output = dtv.run(geometry.bp(sino, dtv.device), sino)
geometry.bp(sino, dtv.device).cpu().numpy().tofile("/media/seu/wyk/Data/raws/r.raw")