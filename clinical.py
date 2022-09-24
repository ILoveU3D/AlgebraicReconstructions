import torch
import numpy as np
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
from Geometry.BeijingGeometry.Geometry import BeijingGeometry
from Algorithm.DTV.CP import DTVCP
from Utils.AstraConvertion import projection2Astra

geometry = BeijingGeometry()
dtv = DTVCP(geometry)

data = np.fromfile("/home/nv/wyk/raws/proj.raw", dtype="float32")
data = np.reshape(data, geometry.torchDetectorSize)
data = torch.from_numpy(data).to(dtv.device)
# geometry.bp(data, "cuda:0").detach().cpu().numpy().tofile("/home/nv/wyk/raws/r.raw")
output = dtv.run(geometry.bp(data, dtv.device), data)

# data = np.fromfile("/home/nv/mcl/data/retrain_real_data/96views/zhiyuanzhe1/label/0.raw", dtype="float32")
# data = np.reshape(data, geometry.torchVolumeSize)
# data = torch.from_numpy(data).to("cuda:0")
# geometry.fp(data, "cuda:0").detach().cpu().numpy().tofile("/home/nv/wyk/raws/r.raw")