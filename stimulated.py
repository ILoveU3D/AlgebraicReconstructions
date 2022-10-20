import torch
import time
import numpy as np
# from Geometry.StandardGeometry.Geometry import StandardGeometry
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
# from Geometry.BeijingGeometry.Geometry import BeijingGeometry

# geometry = BeijingGeometry()
geometry = ShenzhenGeometry()
print("compile finished")
data = np.fromfile("/media/seu/wyk/Data/raws/gM1.raw", dtype="float32")
data = np.reshape(data, [1,1,100,800,800])
# data = np.fromfile("/media/seu/wyk/Data/raws/gM.raw", dtype="float32")
# data = np.reshape(data, [1,1,32,2304,2940])
data = torch.from_numpy(data).to("cuda:0")
tic = time.time()
volume = geometry.fp(data, "cuda:0")
print(time.time()-tic)
volume.detach().to("cpu").numpy().tofile("/media/seu/wyk/Data/raws/sino.raw")