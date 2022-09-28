/**
* 基于标准圆轨迹定义可以用于pytorch深度学习，迭代GPU加速和C++部署的正投影算子
* Author: 姚维国，马春良，王硕然，王煜康
* Note：王煜康
**/
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/helper_math.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 存储体块的纹理内存
texture<float, cudaTextureType3D, cudaReadModeElementType> volumeTexture;

__global__ void forwardKernel(float* sino, float angle,
 const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter, const float sid, const float sdd, const uint index){
    // 像素驱动，此核代表一个探测器像素
    uint2 detectorIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,  blockIdx.y* blockDim.y + threadIdx.y);
    if (detectorIdx.x >= detectorSize.x || detectorIdx.y >= detectorSize.y)
        return;

    // 计算当前角度下的中心射线方向向量与探测器像素的位置坐标
    angle = angle / 180 * PI - PI / 2;
    float2 ex = make_float2(cos(angle), sin(angle));
    float2 ey = make_float2(-ex.y, ex.x);
    float detectorX = detectorIdx.x + detectorCenter.x;
    float detectorY = detectorIdx.y + detectorCenter.y;

    // 计算得到像素射线方向和起始点
    float2 world = ex * sdd + ey * detectorX;
    float3 rayVector = make_float3(world, detectorY);
    rayVector = normalize(rayVector);
    float3 sourcePoint = make_float3(-ex * sid, 0);

    // 计算范围并累加
    float pixel = 0.0f;
    float alpha0, alpha1;
    if (fabs(rayVector.x) >= fabs(rayVector.y)){
        float volume_min_edge_point = volumeCenter.x;
        float volume_max_edge_point = volumeSize.x + volumeCenter.x;
        alpha0 = (volume_min_edge_point - sourcePoint.x) / rayVector.x;
        alpha1 = (volume_max_edge_point - sourcePoint.x) / rayVector.x;
    }
    else{
        float volume_min_edge_point = volumeCenter.y;
        float volume_max_edge_point = volumeSize.y + volumeCenter.y;
        alpha0 = (volume_min_edge_point - sourcePoint.y) / rayVector.y;
        alpha1 = (volume_max_edge_point - sourcePoint.y) / rayVector.y;
    }
    float min_alpha = fmin(alpha0, alpha1) - 3;
    float max_alpha = fmax(alpha0, alpha1) + 3;

    float px, py, pz;
    float sampleInterval = sid / sdd;

    while (min_alpha < max_alpha)
    {
        px = sourcePoint.x + min_alpha * rayVector.x;
        py = sourcePoint.y + min_alpha * rayVector.y;
        pz = sourcePoint.z + min_alpha * rayVector.z;
        px /= sampleInterval;
        py /= sampleInterval;
        pz /= sampleInterval;
        px -= volumeCenter.x;
        py -= volumeCenter.y;
        pz -= volumeCenter.z;
        pixel += tex3D(volumeTexture, px + 0.5f, py + 0.5f, pz + 0.5f);
        min_alpha += sampleInterval;
    }
    pixel *= sampleInterval;

    unsigned sinogramIdx = index * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
    sino[sinogramIdx] = pixel;
}

torch::Tensor forward(torch::Tensor volume, torch::Tensor angles, torch::Tensor _volumeSize, torch::Tensor _detectorSize, const float sid, const float sdd, const long device){
    CHECK_INPUT(volume);
    CHECK_INPUT(angles);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");

    auto out = torch::zeros({volume.size(0), angles.size(0), _detectorSize[1].item<int>(), _detectorSize[0].item<int>()}).to(volume.device());
    float* outPtr = out.data<float>();
    float* volumePtr = volume.data<float>();

    // 初始化纹理
    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    volumeTexture.addressMode[0] = cudaAddressModeBorder;
    volumeTexture.addressMode[1] = cudaAddressModeBorder;
    volumeTexture.addressMode[2] = cudaAddressModeBorder;
    volumeTexture.filterMode = cudaFilterModeLinear;
    volumeTexture.normalized = false;

    // 体块和探测器的大小位置向量化
    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float2 detectorCenter = make_float2(detectorSize) / -2.0;

    for(int batch = 0;batch < volume.size(0); batch++){
        float* volumePtrPitch = volumePtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;
        float* outPtrPitch = outPtr + angles.size(0) * detectorSize.x * detectorSize.y * batch;

        // 绑定纹理
        cudaExtent m_extent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
        cudaArray *volumeArray;
        cudaMalloc3DArray(&volumeArray, &channelDesc, m_extent);
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)volumePtrPitch, volumeSize.x*sizeof(float), volumeSize.x, volumeSize.y);
        copyParams.dstArray = volumeArray;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.extent = m_extent;
        cudaMemcpy3D(&copyParams);
        cudaBindTextureToArray(volumeTexture, volumeArray, channelDesc);

        // 以角度为单位做探测器像素驱动的正投影
        const dim3 blockSize = dim3(BLOCKSIZE_X, BLOCKSIZE_Y, 1 );
        const dim3 gridSize = dim3(detectorSize.x / blockSize.x + 1, detectorSize.y / blockSize.y + 1 , 1);
        for (int angle = 0; angle < angles.size(0); angle++){
           forwardKernel<<<gridSize, blockSize>>>(outPtrPitch, angles[angle].item<float>(), volumeSize, volumeCenter, detectorSize, detectorCenter, sid, sdd, angle);
        }

      // 解绑纹理
      cudaUnbindTexture(volumeTexture);
      cudaFreeArray(volumeArray);
    }
    return out;
}