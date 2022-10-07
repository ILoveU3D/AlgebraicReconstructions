/**
* 基于复眼24系统定义可以用于pytorch深度学习，迭代GPU加速和C++部署的反投影算子
* Author: 姚维国，马春良，王硕然，王煜康
* Note：王煜康
**/
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/helper_math.h"

#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 弦图的纹理内存
texture<float, cudaTextureType3D, cudaReadModeElementType> sinoTexture;

__global__ void backwardKernel(float* volume, float angle, const uint anglesNum, const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter, const float sid, const float sdd, const uint index){
    // 体素驱动，代表一个体素点
   uint2 volumeIdx = make_uint2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);
   if(volumeIdx.x >= volumeSize.x  || volumeIdx.y >= volumeSize.y )
      return;

    // 计算得到探测器像素坐标x,z
   float sampleInterval = sid / sdd;
   const float2 coordinates = make_float2(volumeCenter.x + volumeIdx.x, volumeCenter.y + volumeIdx.y) * sampleInterval;

   angle = angle / 180 * PI - PI / 2;
   float2 ex = make_float2(cos(angle), sin(angle));
   float2 ey = make_float2(-ex.y, ex.x);
   float2 source = ex * sid;
   float2 detector = ex * (sdd - sid);
   float2 intersection = intersectLines2D(coordinates, -1*source, detector, detector + ey);
   float x = dot(intersection, ey) - detectorCenter.x;

   float2 biasRay = source + coordinates;
   float dz = sdd / dot(biasRay, ex) * sampleInterval;
   float z = volumeCenter.z * dz - detectorCenter.y;

    // 计算权重
   float coff = sid / (sid - dot(coordinates, ex));

    // 反投影
   for (int k = 0; k < volumeSize.z; k++){
       int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
       float val = tex3D(sinoTexture, x + 0.5f, z + 0.5f, index+0.5f);
       volume[idx] += val * coff * coff * PI / anglesNum;
       z += dz;
   }
}

torch::Tensor backward(torch::Tensor sino, torch::Tensor angles, torch::Tensor _volumeSize, torch::Tensor _detectorSize, const float sid, const float sdd, const int device){
    CHECK_INPUT(sino);
    CHECK_INPUT(angles);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");

    auto out = torch::zeros({sino.size(0), _volumeSize[2].item<int>(), _volumeSize[1].item<int>(), _volumeSize[0].item<int>()}).to(sino.device());
    float* outPtr = out.data<float>();
    float* sinoPtr = sino.data<float>();

    // 初始化纹理
    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    sinoTexture.addressMode[0] = cudaAddressModeBorder;
    sinoTexture.addressMode[1] = cudaAddressModeBorder;
    sinoTexture.addressMode[2] = cudaAddressModeBorder;
    sinoTexture.filterMode = cudaFilterModeLinear;
    sinoTexture.normalized = false;

    // 体块和探测器的大小位置向量化
    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float2 detectorCenter = make_float2(detectorSize) / -2.0;

    for(int batch = 0;batch < sino.size(0); batch++){
        float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles.size(0) * batch;
        float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

        // 绑定纹理
        cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles.size(0));
        cudaArray *sinoArray;
        cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x*sizeof(float), detectorSize.x, detectorSize.y);
        copyParams.dstArray = sinoArray;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.extent = m_extent;
        cudaMemcpy3D(&copyParams);
        cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

        // 以角度为单位做体素驱动的反投影
        const dim3 blockSize = dim3(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
        const dim3 gridSize = dim3(volumeSize.x / blockSize.x + 1, volumeSize.y / blockSize.y + 1 , 1);
        for (int angle = 0; angle < angles.size(0); angle++){
           backwardKernel<<<gridSize, blockSize>>>(outPtrPitch, angles[angle].item<float>(), angles.size(0), volumeSize, volumeCenter, detectorSize, detectorCenter, sid, sdd, angle);
        }

      // 解绑纹理
      cudaUnbindTexture(sinoTexture);
      cudaFreeArray(sinoArray);
    }
    return out;
}