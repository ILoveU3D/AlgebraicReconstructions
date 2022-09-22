import os
import numpy as np

path = r"/media/seu/wyk/Data/raws/0"
temp = np.zeros([72,3072,288], dtype="float32")
with open(os.path.join(path, "..", "files.txt"),'r') as f:
    for i,name in enumerate(f.readlines()):
        r = np.fromfile(os.path.join(path, name.rstrip('\n')), "float32")
        temp[i,:,:] = r.reshape([3072,288])
temp.tofile(r"/media/seu/wyk/Data/raws/proj.raw")