import torch
import time
import numpy as np
# from Geometry.StandardGeometry.Geometry import StandardGeometry
from Geometry.ShenzhenGeometry.GeometryFly import ShenzhenGeometry
# from Geometry.BeijingGeometry.Geometry import BeijingGeometry

if __name__ == '__main__':
    # geometry = BeijingGeometry()
    geometry = ShenzhenGeometry()
    print("compile finished")
    data = np.fromfile("/home/nv/wyk/raws/shenzhendata2/停拍/不同位置/output/result_700_102376472576.0.raw", dtype="float32")
    data = np.reshape(data, [1,1,100,800,800])
    # data = np.fromfile("/home/nv/wyk/raws/temp/65views_21.55.raw", dtype="float32")
    # data = np.reshape(data, [1,1,32,2304,2940])
    data = torch.from_numpy(data).to("cuda:0")
    tic = time.time()
    volume = geometry.fp(data, "cuda:0")
    print(time.time()-tic)
    volume.detach().to("cpu").numpy().tofile("/home/nv/wyk/raws/sino.raw")