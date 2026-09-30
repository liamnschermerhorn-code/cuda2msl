#pragma once

// C++ declarations for Objective-C++ MPS bridge functions
// These use void* for Objective-C types so pure C++ can call them

#include "cuda.h"
#include "cublas_v2.h"

namespace cuda_metal {

// GEMM via MPSMatrixMultiplication
// cmdBuf, bufferA, bufferB, bufferC are Objective-C id<MTLCommandBuffer>/id<MTLBuffer>
// cast to void* for C++ compatibility
cublasStatus_t mps_gemm_f32(
    void* cmdBuf,       // id<MTLCommandBuffer>
    bool transposeA, bool transposeB,
    int M, int N, int K,
    float alpha,
    void* bufferA, size_t offsetA, int ldA,
    void* bufferB, size_t offsetB, int ldB,
    float beta,
    void* bufferC, size_t offsetC, int ldC);

cublasStatus_t mps_gemm_f16(
    void* cmdBuf,
    bool transposeA, bool transposeB,
    int M, int N, int K,
    float alpha,
    void* bufferA, size_t offsetA, int ldA,
    void* bufferB, size_t offsetB, int ldB,
    float beta,
    void* bufferC, size_t offsetC, int ldC);

// Mixed precision GEMM: fp16 inputs, fp32 output, fp32 accumulation
cublasStatus_t mps_gemm_f16_f32(
    void* cmdBuf,
    bool transposeA, bool transposeB,
    int M, int N, int K,
    float alpha,
    void* bufferA, size_t offsetA, int ldA,   // fp16
    void* bufferB, size_t offsetB, int ldB,   // fp16
    float beta,
    void* bufferC, size_t offsetC, int ldC);  // fp32

// Mixed precision GEMM: bf16 inputs, fp32 output, fp32 accumulation
cublasStatus_t mps_gemm_bf16_f32(
    void* cmdBuf,
    bool transposeA, bool transposeB,
    int M, int N, int K,
    float alpha,
    void* bufferA, size_t offsetA, int ldA,   // bf16
    void* bufferB, size_t offsetB, int ldB,   // bf16
    float beta,
    void* bufferC, size_t offsetC, int ldC);  // fp32

// Batched GEMM via MPS: C[b] = alpha * A[b] * B[b] + beta * C[b]
cublasStatus_t mps_batched_gemm_f16(
    void* cmdBuf,
    bool transposeA, bool transposeB,
    int M, int N, int K,
    float alpha,
    void* bufferA, size_t offsetA, int ldA, size_t strideA,
    void* bufferB, size_t offsetB, int ldB, size_t strideB,
    float beta,
    void* bufferC, size_t offsetC, int ldC, size_t strideC,
    int batchCount);

// Conv2D via MPSCNNConvolution
// input: NCHW, weight: OIHW, output: NCHW
cublasStatus_t mps_conv2d_f32(
    void* cmdBuf,
    void* input,  size_t inputOffset,
    void* weight, size_t weightOffset,
    void* bias,   size_t biasOffset,   // nullable
    void* output, size_t outputOffset,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int padH, int padW,
    int strideH, int strideW);

// Dispatch router: choose MPS vs Metal compute based on problem size.
// Returns true if MPS path was used, false if caller should use compute.
bool mps_should_use_gemm(int M, int N, int K);

// Create a Metal buffer wrapping existing memory (zero-copy).
// Pointer must be page-aligned (e.g., from mmap).
// Returns the MTL::Buffer* as void*, or nullptr on failure.
void* metal_wrap_existing_buffer(void* ptr, size_t size);

} // namespace cuda_metal
