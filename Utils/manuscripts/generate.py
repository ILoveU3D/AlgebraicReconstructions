import os
import numpy as np

path = r"/home/nv/mcl/data/lung/zhiyuanzhe/corr/0_binning"
temp = np.zeros([360,72,769], dtype="float32")
with open(os.path.join(path, "..", "files.txt"),'r') as f:
    for i,name in enumerate(f.readlines()):
        r = np.fromfile(os.path.join(path, name.rstrip('\n')), "float32")
        temp[i,:,:] = r.reshape([72,769])
temp.tofile(r"/home/nv/wyk/raws/proj.raw")