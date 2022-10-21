#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/helper_math.h"
#include "../include/helper_geometry.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_Z 4
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 存储体块的纹理内存
texture<float, cudaTextureType3D, cudaReadModeElementType> volumeTexture;

__global__ void forwardKernel(float* sino, const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter, const float* projectVector, const uint index, const uint systemNum){
    __shared__ float cache[BLOCK_Z];
    // 像素驱动，此核代表一个探测器像素
    uint3 detectorIdx = make_uint3(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z*blockDim.z+threadIdx.z);
    if (detectorIdx.x >= detectorSize.x || detectorIdx.y >= detectorSize.y || detectorIdx.z >= systemNum){
        return;
    }

    float detectorX = detectorIdx.x + detectorCenter.x;
    float detectorY = detectorIdx.y + detectorCenter.y;

    unsigned projectVectorIdx = detectorIdx.z * 12;
    float3 sourcePosition = make_float3(projectVector[projectVectorIdx], projectVector[projectVectorIdx+1], projectVector[projectVectorIdx+2]);
    float3 detectorPosition = make_float3(projectVector[projectVectorIdx+3], projectVector[projectVectorIdx+4], projectVector[projectVectorIdx+5]);
    float3 u = make_float3(projectVector[projectVectorIdx+6], projectVector[projectVectorIdx+7], projectVector[projectVectorIdx+8]);
    float3 v = make_float3(projectVector[projectVectorIdx+9], projectVector[projectVectorIdx+10], projectVector[projectVectorIdx+11]);

    // 计算当前角度下的中心射线方向向量与探测器像素的位置坐标
    float3 detectorPixel = detectorPosition + (0.5f+detectorX) *u + (0.5f+detectorY) * v ;
    // 计算得到像素射线方向和起始点
    float3 sourcePoint = sourcePosition;
    float3 rayVector = detectorPixel-sourcePoint;
    rayVector = normalize(rayVector);

    // 计算范围并累加
    float pixel = 0.0f;
    float alpha0, alpha1;
    float rayVectorDomainDim=fmax(fabs(rayVector.x),fmax(fabs(rayVector.z),fabs(rayVector.y)));

    if (fabs(rayVector.x) == rayVectorDomainDim){
        float volume_min_edge_point = volumeCenter.x;
        float volume_max_edge_point = volumeSize.x + volumeCenter.x;
        alpha0 = (volume_min_edge_point - sourcePoint.x) / rayVector.x;
        alpha1 = (volume_max_edge_point - sourcePoint.x) / rayVector.x;
    }
    else if(fabs(rayVector.y) == rayVectorDomainDim){
        float volume_min_edge_point = volumeCenter.y;
        float volume_max_edge_point = volumeSize.y + volumeCenter.y;
        alpha0 = (volume_min_edge_point - sourcePoint.y) / rayVector.y;
        alpha1 = (volume_max_edge_point - sourcePoint.y) / rayVector.y;
    }
    else {
        float volume_min_edge_point = volumeCenter.z;
        float volume_max_edge_point = volumeSize.z + volumeCenter.z;
        alpha0 = (volume_min_edge_point - sourcePoint.z) / rayVector.z;
        alpha1 = (volume_max_edge_point - sourcePoint.z) / rayVector.z;
    }

    float min_alpha = fmin(alpha0, alpha1) - 3;
    float max_alpha = fmax(alpha0, alpha1) + 3;

    float px, py, pz;
    float sampleInterval = fabs(sourcePosition.z) / fabs(detectorPixel.z-sourcePosition.z);

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
    pixel *= sampleInterval / systemNum;
    cache[threadIdx.z] = pixel;
    sum(cache, threadIdx.z);
    if(threadIdx.z == 0){
        unsigned sinogramIdx = index * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
        atomicAdd(&sino[sinogramIdx], cache[0]);
//    sino[sinogramIdx] = pixel;
    }
}

torch::Tensor forward(torch::Tensor volume, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector, const int systemNum, const long device){
    CHECK_INPUT(volume);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0) / systemNum;
    auto out = torch::zeros({volume.size(0), angles, _detectorSize[1].item<int>(), _detectorSize[0].item<int>()}).to(volume.device());
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
        float* outPtrPitch = outPtr + angles * detectorSize.x * detectorSize.y * batch;

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
        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);
        const dim3 gridSize = dim3(detectorSize.x / BLOCK_X + 1, detectorSize.y / BLOCK_Y + 1, systemNum / BLOCK_Z + 1);
        printf("thread:(%d %d %d)\n",blockSize.x,blockSize.y,blockSize.z);
        printf("block:(%d %d %d)\n",gridSize.x,gridSize.y,gridSize.z);
        auto projVec = projectVector.reshape({angles, systemNum * 12});
        for (int angle = 0; angle < angles; angle++){
           forwardKernel<<<gridSize, blockSize>>>(outPtrPitch, volumeSize, volumeCenter, detectorSize, detectorCenter, (float*)projVec[angle].data<float>(), angle, systemNum);
        }

      // 解绑纹理
      cudaUnbindTexture(volumeTexture);
      cudaFreeArray(volumeArray);
    }
    return out;
}