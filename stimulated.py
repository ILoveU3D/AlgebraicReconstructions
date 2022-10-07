import torch
import numpy as np
# from Geometry.ShenzhenGeometry.GeometryStationary import ShenzhenGeometry
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
from Geometry.StandardGeometry.Geometry import StandardGeometry
# from Geometry.BeijingGeometry.Geometry import BeijingGeometry
from Algorithm.DTV.CP import DTVCP

geometry = StandardGeometry()
dtv = DTVCP(geometry)
print("compile finished")
data = np.fromfile("/media/seu/wyk/Data/raws/trainData/1.raw", dtype="float32")
data = np.reshape(data, [1,1,16,256,256])
data = torch.from_numpy(data).to(dtv.device)
sino = geometry.fp(data, dtv.device)
dtv.run(geometry.bp(sino, dtv.device), sino)
# geometry.bp(sino, dtv.device).detach().cpu().numpy().tofile("/media/seu/wyk/Data/raws/r.raw")