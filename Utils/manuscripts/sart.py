import torch
import time
import numpy as np
from tqdm import trange
from Geometry.ShenzhenGeometry.GeometryStationary import ShenzhenGeometry

if __name__ == '__main__':
    geometry = ShenzhenGeometry()
    device = "cuda:0"
    data = np.fromfile("/home/nv/wyk/raws/proj.raw", dtype="float32")
    data = np.reshape(data, geometry.torchDetectorSize)
    data = torch.from_numpy(data).to(device)
    projection = geometry.fp(data)
    r = torch.zeros(geometry.torchVolumeSize).to(device)
    for i in trange(100):
        r += 0.1 * geometry.bp(data - geometry.fp(r, device), device)
    r.detach().cpu().numpy().tofile("/home/nv/wyk/raws/r.raw")