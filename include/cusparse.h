#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's cusparse.h

#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// cuSPARSE status codes
// ---------------------------------------------------------------------------

typedef enum {
    CUSPARSE_STATUS_SUCCESS                   = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED           = 1,
    CUSPARSE_STATUS_ALLOC_FAILED              = 2,
    CUSPARSE_STATUS_INVALID_VALUE             = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH             = 4,
    CUSPARSE_STATUS_MAPPING_ERROR             = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED          = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR            = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT                = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED             = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11
} cusparseStatus_t;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE       = 0,
    CUSPARSE_OPERATION_TRANSPOSE           = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

typedef enum {
    CUSPARSE_INDEX_16U = 1,
    CUSPARSE_INDEX_32I = 2,
    CUSPARSE_INDEX_64I = 3
} cusparseIndexType_t;

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0,
    CUSPARSE_INDEX_BASE_ONE  = 1
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_ORDER_COL = 1,
    CUSPARSE_ORDER_ROW = 2
} cusparseOrder_t;

// Data types (matching NVIDIA cudaDataType values)
typedef enum {
    CUDA_R_32F_CUSPARSE = 0,
    CUDA_R_64F_CUSPARSE = 1,
    CUDA_R_16F_CUSPARSE = 2
} cusparseDataType_t;

// Use the same cudaDataType values as NVIDIA
#ifndef CUDA_R_32F
#define CUDA_R_32F 0
#endif
#ifndef CUDA_R_64F
#define CUDA_R_64F 1
#endif
#ifndef CUDA_R_16F
#define CUDA_R_16F 2
#endif

// SpMV algorithm selection
typedef enum {
    CUSPARSE_SPMV_ALG_DEFAULT    = 0,
    CUSPARSE_SPMV_CSR_ALG1       = 1,
    CUSPARSE_SPMV_CSR_ALG2       = 2
} cusparseSpMVAlg_t;

// SpMM algorithm selection
typedef enum {
    CUSPARSE_SPMM_ALG_DEFAULT    = 0,
    CUSPARSE_SPMM_CSR_ALG1       = 1,
    CUSPARSE_SPMM_CSR_ALG2       = 2,
    CUSPARSE_SPMM_CSR_ALG3       = 3
} cusparseSpMMAlg_t;

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

typedef struct cusparseContext*    cusparseHandle_t;
typedef struct cusparseMatDescr*   cusparseMatDescr_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseDnMatDescr* cusparseDnMatDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;

// ---------------------------------------------------------------------------
// Handle management
// ---------------------------------------------------------------------------

cusparseStatus_t cusparseCreate(cusparseHandle_t* handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t stream);
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* stream);

// ---------------------------------------------------------------------------
// Matrix descriptor (legacy)
// ---------------------------------------------------------------------------

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA);
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA);

// ---------------------------------------------------------------------------
// Sparse matrix descriptor (CSR) - generic API
// ---------------------------------------------------------------------------

cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                                   int64_t rows,
                                   int64_t cols,
                                   int64_t nnz,
                                   void* csrRowOffsets,
                                   void* csrColInd,
                                   void* csrValues,
                                   cusparseIndexType_t csrRowOffsetsType,
                                   cusparseIndexType_t csrColIndType,
                                   cusparseIndexBase_t idxBase,
                                   int cudaDataType);

cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr);

// ---------------------------------------------------------------------------
// Dense matrix descriptor - generic API
// ---------------------------------------------------------------------------

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                                     int64_t rows,
                                     int64_t cols,
                                     int64_t ld,
                                     void* values,
                                     int cudaDataType,
                                     cusparseOrder_t order);

cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr);

// ---------------------------------------------------------------------------
// Dense vector descriptor - generic API
// ---------------------------------------------------------------------------

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                                     int64_t size,
                                     void* values,
                                     int cudaDataType);

cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr);

// ---------------------------------------------------------------------------
// Sparse Matrix-Vector Multiplication (SpMV)
//   y = alpha * op(A) * x + beta * y
// ---------------------------------------------------------------------------

cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle,
                                         cusparseOperation_t opA,
                                         const void* alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnVecDescr_t vecX,
                                         const void* beta,
                                         cusparseDnVecDescr_t vecY,
                                         int computeType,
                                         cusparseSpMVAlg_t alg,
                                         size_t* bufferSize);

cusparseStatus_t cusparseSpMV(cusparseHandle_t handle,
                              cusparseOperation_t opA,
                              const void* alpha,
                              cusparseSpMatDescr_t matA,
                              cusparseDnVecDescr_t vecX,
                              const void* beta,
                              cusparseDnVecDescr_t vecY,
                              int computeType,
                              cusparseSpMVAlg_t alg,
                              void* externalBuffer);

// ---------------------------------------------------------------------------
// Sparse Matrix-Dense Matrix Multiplication (SpMM)
//   C = alpha * op(A) * B + beta * C
// ---------------------------------------------------------------------------

cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle,
                                         cusparseOperation_t opA,
                                         cusparseOperation_t opB,
                                         const void* alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseDnMatDescr_t matB,
                                         const void* beta,
                                         cusparseDnMatDescr_t matC,
                                         int computeType,
                                         cusparseSpMMAlg_t alg,
                                         size_t* bufferSize);

cusparseStatus_t cusparseSpMM(cusparseHandle_t handle,
                              cusparseOperation_t opA,
                              cusparseOperation_t opB,
                              const void* alpha,
                              cusparseSpMatDescr_t matA,
                              cusparseDnMatDescr_t matB,
                              const void* beta,
                              cusparseDnMatDescr_t matC,
                              int computeType,
                              cusparseSpMMAlg_t alg,
                              void* externalBuffer);

#ifdef __cplusplus
}
#endif
