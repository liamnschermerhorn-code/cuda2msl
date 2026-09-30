#pragma once

// CUDA-to-Metal Translation Layer
// Core type definitions matching NVIDIA CUDA driver API types

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum cudaError {
    cudaSuccess                    = 0,
    cudaErrorInvalidValue          = 1,
    cudaErrorMemoryAllocation      = 2,
    cudaErrorInitializationError   = 3,
    cudaErrorInvalidDevicePointer  = 17,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorInvalidDevice         = 101,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNotReady              = 600,
    cudaErrorUnknown               = 999
} cudaError_t;

// Memory copy direction
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4
} cudaMemcpyKind;

// Opaque handle types
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st*  cudaEvent_t;

// dim3 - grid/block dimensions
struct dim3 {
    unsigned int x, y, z;
#ifdef __cplusplus
    constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
#endif
};

// Device properties structure
struct cudaDeviceProp {
    char     name[256];
    size_t   totalGlobalMem;
    size_t   sharedMemPerBlock;
    int      regsPerBlock;
    int      warpSize;
    size_t   memPitch;
    int      maxThreadsPerBlock;
    int      maxThreadsDim[3];
    int      maxGridSize[3];
    int      clockRate;
    size_t   totalConstMem;
    int      major;
    int      minor;
    size_t   textureAlignment;
    size_t   texturePitchAlignment;
    int      deviceOverlap;
    int      multiProcessorCount;
    int      kernelExecTimeoutEnabled;
    int      integrated;
    int      canMapHostMemory;
    int      computeMode;
    int      maxTexture1D;
    int      maxTexture1DMipmap;
    int      maxTexture1DLinear;
    int      maxTexture2D[2];
    int      maxTexture2DMipmap[2];
    int      maxTexture2DLinear[3];
    int      maxTexture2DGather[2];
    int      maxTexture3D[3];
    int      maxTexture3DAlt[3];
    int      maxTextureCubemap;
    int      maxTexture1DLayered[2];
    int      maxTexture2DLayered[3];
    int      maxTextureCubemapLayered[2];
    int      maxSurface1D;
    int      maxSurface2D[2];
    int      maxSurface3D[3];
    int      maxSurface1DLayered[2];
    int      maxSurface2DLayered[3];
    int      maxSurfaceCubemap;
    int      maxSurfaceCubemapLayered[2];
    size_t   surfaceAlignment;
    int      concurrentKernels;
    int      ECCEnabled;
    int      pciBusID;
    int      pciDeviceID;
    int      pciDomainID;
    int      tccDriver;
    int      asyncEngineCount;
    int      unifiedAddressing;
    int      memoryClockRate;
    int      memoryBusWidth;
    int      l2CacheSize;
    int      persistingL2CacheMaxSize;
    int      maxThreadsPerMultiProcessor;
    int      streamPrioritiesSupported;
    int      globalL1CacheSupported;
    int      localL1CacheSupported;
    size_t   sharedMemPerMultiprocessor;
    int      regsPerMultiprocessor;
    int      managedMemory;
    int      isMultiGpuBoard;
    int      multiGpuBoardGroupID;
    int      singleToDoublePrecisionPerfRatio;
    int      pageableMemoryAccess;
    int      concurrentManagedAccess;
    int      computePreemptionSupported;
    int      canUseHostPointerForRegisteredMem;
    int      cooperativeLaunch;
    int      cooperativeMultiDeviceLaunch;
    int      pageableMemoryAccessUsesHostPageTables;
    int      directManagedMemAccessFromHost;
    int      accessPolicyMaxWindowSize;
};

#ifdef __cplusplus
}
#endif
