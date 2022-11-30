#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/helper_math.h"
#include "../include/helper_geometry.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 弦图的纹理内存
texture<float, cudaTextureType3D, cudaReadModeElementType> sinoTexture;

__global__ void backwardKernel(float* volume, const uint3 volumeSize, const uint2 detectorSize, const float* projectVector, const uint index, const uint anglesNum,const float3 volumeCenter, const float2 detectorCenter){
    // 体素驱动，代表一个体素点
    uint3 volumeIdx = make_uint3(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z*blockDim.z+threadIdx.z);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y){
        return;
    }

    unsigned projectVectorIdx = volumeIdx.z * 12;
    float3 sourcePosition = make_float3(projectVector[projectVectorIdx], projectVector[projectVectorIdx+1], projectVector[projectVectorIdx+2]);
    float3 detectorPosition = make_float3(projectVector[projectVectorIdx+3], projectVector[projectVectorIdx+4], projectVector[projectVectorIdx+5]);
    float3 u = make_float3(projectVector[projectVectorIdx+6], projectVector[projectVectorIdx+7], projectVector[projectVectorIdx+8]);
    float3 v = make_float3(projectVector[projectVectorIdx+9], projectVector[projectVectorIdx+10], projectVector[projectVectorIdx+11]);

    float sampleInterval = fabs(sourcePosition.z)/fabs(sourcePosition.z-detectorPosition.z);
    for (int k = 0;k < volumeSize.z;k++)
    {
        const float3 coordinates = make_float3(volumeCenter.x + volumeIdx.x, volumeCenter.y + volumeIdx.y,volumeCenter.z+k);
        float3 normVector = cross(u,v);
        float3 intersection = intersectLines3D(sourcePosition,coordinates,detectorPosition,detectorPosition+normVector);
        float detectorX = dot(intersection-detectorPosition, u) - detectorCenter.x;
        float detectorY = dot(intersection-detectorPosition, v) - detectorCenter.y;
        int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
        float val = tex3D(sinoTexture, detectorX + 0.5f, detectorY + 0.5f, index+0.5f);
        volume[idx] += val * 2 * PI / anglesNum;
    }
}

torch::Tensor backward(torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector, const long device){
    CHECK_INPUT(sino);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0);
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
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    float2 detectorCenter = make_float2(detectorSize) / -2.0;
    for(int batch = 0;batch < sino.size(0); batch++){
        float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
        float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

        // 绑定纹理
        cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
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
        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
        const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, 1);
        printf("thread:(%d %d %d)\n",blockSize.x,blockSize.y,blockSize.z);
        printf("block:(%d %d %d)\n",gridSize.x,gridSize.y,gridSize.z);
        for (int angle = 0; angle < angles; angle++){
           backwardKernel<<<gridSize, blockSize>>>(outPtrPitch, volumeSize, detectorSize, (float*)projectVector[angle].data<float>(), angle, angles, volumeCenter, detectorCenter);
        }

      // 解绑纹理
      cudaUnbindTexture(sinoTexture);
      cudaFreeArray(sinoArray);
    }
    return out;
}