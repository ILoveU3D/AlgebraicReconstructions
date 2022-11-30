import torch
import time
import numpy as np
# from Geometry.StandardGeometry.Geometry import StandardGeometry
from Geometry.ShenzhenGeometry.Geometry import ShenzhenGeometry
from Geometry.BeijingGeometry.Geometry import BeijingGeometry

if __name__ == '__main__':
    geometry = BeijingGeometry()
    # geometry = ShenzhenGeometry()
    print("compile finished")
    # sino = np.fromfile("/media/seu/wyk/Data/raws/sino.raw", dtype="float32")
    # sino = np.reshape(sino, [1,1,32,2304,2940])
    data = np.fromfile("/media/seu/wyk/beijing/sample.raw", dtype="float32")
    data = np.reshape(data, [1,1,72,256,256])
    # sino = torch.from_numpy(sino).to("cuda:0")
    # tic = time.time()
    # volume = geometry.bp(sino, "cuda:0")
    # print(time.time() - tic)
    # volume.detach().to("cpu").numpy().tofile("/media/seu/wyk/Data/raws/volume.raw")
    tic = time.time()
    volume = torch.from_numpy(data).cuda()
    gM = geometry.fp(volume, "cuda:0")
    print(time.time()-tic)
    gM.detach().to("cpu").numpy().tofile("/media/seu/wyk/Data/raws/gM.raw")
    volume = geometry.bp(gM,"cuda:0")
    volume.detach().to("cpu").numpy().tofile("/media/seu/wyk/Data/raws/volume.raw")
    print(time.time() - tic)