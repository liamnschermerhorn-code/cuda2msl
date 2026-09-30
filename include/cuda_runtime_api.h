#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's cuda_runtime_api.h

#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Device Management
// ---------------------------------------------------------------------------

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceReset(void);

// ---------------------------------------------------------------------------
// Memory Management
// ---------------------------------------------------------------------------

cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count,
                            cudaStream_t stream);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);

// ---------------------------------------------------------------------------
// Stream Management
// ---------------------------------------------------------------------------

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream,
                                      unsigned int flags);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaDeviceSynchronize(void);

// Stream creation flags
#define cudaStreamDefault     0x00
#define cudaStreamNonBlocking 0x01

// ---------------------------------------------------------------------------
// Event Management
// ---------------------------------------------------------------------------

cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start,
                                 cudaEvent_t stop);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);

// Event creation flags
#define cudaEventDefault       0x00
#define cudaEventBlockingSync  0x01
#define cudaEventDisableTiming 0x02

// ---------------------------------------------------------------------------
// Kernel Launch
// ---------------------------------------------------------------------------

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// Error Handling
// ---------------------------------------------------------------------------

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorString(cudaError_t error);
const char* cudaGetErrorName(cudaError_t error);

#ifdef __cplusplus
}
#endif
