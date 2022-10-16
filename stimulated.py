import torch
import time
import numpy as np
# from Geometry.StandardGeometry.Geometry import StandardGeometry
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
# from Geometry.BeijingGeometry.Geometry import BeijingGeometry

# geometry = BeijingGeometry()
geometry = ShenzhenGeometry()
print("compile finished")
data = np.fromfile("/media/seu/wyk/shenzhen/gM1.raw", dtype="float32")
data = np.reshape(data, [1,1,100,800,800])
data = torch.from_numpy(data).to("cuda:0")
tic = time.time()
sino = geometry.fp(data, "cuda:0")
print(time.time()-tic)
sino.detach().to("cpu").numpy().tofile("/media/seu/wyk/Data/raws/sino.raw")
# geometry.bp(sino, "cuda:0").detach().cpu().numpy().tofile("/media/seu/wyk/Data/raws/r.raw")