# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:55:14 2022

@author: Chunliang Ma
"""

import numpy as np
import os

def bilinear_interpolation(img, out_dim):
    channel, src_h, src_w = img.shape  # 原图片的高、宽、通道数
    dst_h, dst_w = out_dim[0], out_dim[1]  # 输出图片的高、宽
    #print('src_h,src_w=', src_h, src_w)
    #print('dst_h,dst_w=', dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((channel, dst_h, dst_w), dtype=np.float32)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):  # 指定 通道数，对channel循环
        for dst_y in range(dst_h):  # 指定 高，对height循环
            for dst_x in range(dst_w):  # 指定 宽，对width循环

                # 源图像和目标图像几何中心的对齐
                # src_x = (dst_x + 0.5) * srcWidth/dstWidth - 0.5
                # src_y = (dst_y + 0.5) * srcHeight/dstHeight - 0.5
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 计算在源图上四个近邻点的位置
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 双线性插值
                temp0 = (src_x1 - src_x) * img[i, src_y0, src_x0] + (src_x - src_x0) * img[i, src_y0, src_x1]
                temp1 = (src_x1 - src_x) * img[i, src_y1, src_x0] + (src_x - src_x0) * img[i, src_y1, src_x1]
                dst_img[i, dst_y, dst_x] = (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1

    return dst_img
#https://blog.csdn.net/weixin_44359479/article/details/123794413
"""
input_raw = np.fromfile("/home/nv/mcl/data/retrain_real_data/focal_shift/Origin2/SS_F0_457.05215_0.00000_01_0217_1166_0000_0000.raw",dtype=np.float32).reshape(1,288,3072)
input_raw = bilinear_interpolation(input_raw, (36,384))
input_raw.astype(np.float32).tofile("/home/nv/mcl/data/retrain_real_data/focal_shift/SS_F0_457.05215_0.00000_01_0217_1166_0000_0000.raw")
"""
input_root = "/home/nv/mcl/data/lung/head/1"
save_root = "/home/nv/mcl/data/lung/head/binning"
if not os.path.exists(save_root):
    os.makedirs(save_root)
input_dir = os.listdir(input_root)
for name in input_dir:
    print(name)
    input_path = os.path.join(input_root, name)
    data = np.fromfile(input_path, dtype="float32")
    data = np.reshape(data, [1, 288, 3072])
    data = bilinear_interpolation(data, [72, 769])
    data.astype(np.float32).tofile(os.path.join(save_root, name))
print("Finish!")
