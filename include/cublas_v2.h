#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's cublas_v2.h

#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// cuBLAS status codes
typedef enum {
    CUBLAS_STATUS_SUCCESS          = 0,
    CUBLAS_STATUS_NOT_INITIALIZED  = 1,
    CUBLAS_STATUS_ALLOC_FAILED     = 3,
    CUBLAS_STATUS_INVALID_VALUE    = 7,
    CUBLAS_STATUS_ARCH_MISMATCH    = 8,
    CUBLAS_STATUS_MAPPING_ERROR    = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR   = 14,
    CUBLAS_STATUS_NOT_SUPPORTED    = 15
} cublasStatus_t;

// Operation types
typedef enum {
    CUBLAS_OP_N = 0,  // No transpose
    CUBLAS_OP_T = 1,  // Transpose
    CUBLAS_OP_C = 2   // Conjugate transpose
} cublasOperation_t;

// Fill mode (for triangular/symmetric operations)
typedef enum {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL  = 2
} cublasFillMode_t;

// Data types for GemmEx
typedef enum {
    CUDA_R_16F  = 2,   // half
    CUDA_R_32F  = 0,   // float
    CUDA_R_64F  = 1,   // double
    CUDA_R_16BF = 14   // bfloat16
} cudaDataType_t;

// GemmEx algorithm (ignored — MPS chooses its own)
typedef enum {
    CUBLAS_GEMM_DEFAULT = -1
} cublasGemmAlgo_t;

// Opaque handle
typedef struct cublasContext* cublasHandle_t;

// Handle management
cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream);

// GEMM - Single precision
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float* alpha,
                           const float* A, int lda,
                           const float* B, int ldb,
                           const float* beta,
                           float* C, int ldc);

// GEMM - Half precision
cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const void* alpha,   // __half*
                           const void* A, int lda,
                           const void* B, int ldb,
                           const void* beta,    // __half*
                           void* C, int ldc);

// GEMM - Extended precision (mixed types)
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void* alpha,
                            const void* A, cudaDataType_t Atype, int lda,
                            const void* B, cudaDataType_t Btype, int ldb,
                            const void* beta,
                            void* C, cudaDataType_t Ctype, int ldc,
                            cudaDataType_t computeType,
                            cublasGemmAlgo_t algo);

// AXPY - Single precision: y = alpha * x + y
cublasStatus_t cublasSaxpy(cublasHandle_t handle,
                           int n,
                           const float* alpha,
                           const float* x, int incx,
                           float* y, int incy);

// SCAL - Single precision: x = alpha * x
cublasStatus_t cublasSscal(cublasHandle_t handle,
                           int n,
                           const float* alpha,
                           float* x, int incx);

// Batched GEMM - Single precision
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float* beta,
    float* C, int ldc, long long int strideC,
    int batchCount);

#ifdef __cplusplus
}
#endif
