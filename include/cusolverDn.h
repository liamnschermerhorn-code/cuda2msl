#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's cusolverDn.h (dense solver)

#include "cuda_runtime_api.h"
#include "cublas_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

// cuSOLVER status codes
typedef enum {
    CUSOLVER_STATUS_SUCCESS                   = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED           = 1,
    CUSOLVER_STATUS_ALLOC_FAILED              = 2,
    CUSOLVER_STATUS_INVALID_VALUE             = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH             = 4,
    CUSOLVER_STATUS_MAPPING_ERROR             = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED          = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR            = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED             = 9,
    CUSOLVER_STATUS_ZERO_PIVOT                = 10,
    CUSOLVER_STATUS_INVALID_LICENSE           = 11
} cusolverStatus_t;

// Eigenvalue job type
typedef enum {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,  // Eigenvalues only
    CUSOLVER_EIG_MODE_VECTOR  = 1   // Eigenvalues and eigenvectors
} cusolverEigMode_t;

// Opaque handle
typedef struct cusolverDnContext* cusolverDnHandle_t;

// ---------------------------------------------------------------------------
// Handle management
// ---------------------------------------------------------------------------

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle,
                                     cudaStream_t stream);
cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle,
                                     cudaStream_t* stream);

// ---------------------------------------------------------------------------
// LU factorization (Sgetrf / Sgetrs)
// ---------------------------------------------------------------------------

cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle,
                                             int m, int n,
                                             float* A, int lda,
                                             int* lwork);

cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle,
                                  int m, int n,
                                  float* A, int lda,
                                  float* workspace,
                                  int* ipiv,
                                  int* info);

cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle,
                                  cublasOperation_t trans,
                                  int n, int nrhs,
                                  const float* A, int lda,
                                  const int* ipiv,
                                  float* B, int ldb,
                                  int* info);

// ---------------------------------------------------------------------------
// Cholesky factorization (Spotrf / Spotrs)
// ---------------------------------------------------------------------------

cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo,
                                             int n,
                                             float* A, int lda,
                                             int* lwork);

cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n,
                                  float* A, int lda,
                                  float* workspace, int lwork,
                                  int* info);

cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo,
                                  int n, int nrhs,
                                  const float* A, int lda,
                                  float* B, int ldb,
                                  int* info);

// ---------------------------------------------------------------------------
// QR factorization (Sgeqrf)
// ---------------------------------------------------------------------------

cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle,
                                             int m, int n,
                                             float* A, int lda,
                                             int* lwork);

cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle,
                                  int m, int n,
                                  float* A, int lda,
                                  float* tau,
                                  float* workspace, int lwork,
                                  int* info);

// ---------------------------------------------------------------------------
// SVD (Sgesvd)
// ---------------------------------------------------------------------------

cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle,
                                             int m, int n,
                                             int* lwork);

cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle,
                                  signed char jobu, signed char jobvt,
                                  int m, int n,
                                  float* A, int lda,
                                  float* S,
                                  float* U, int ldu,
                                  float* VT, int ldvt,
                                  float* workspace, int lwork,
                                  float* rwork,
                                  int* info);

// ---------------------------------------------------------------------------
// Symmetric eigenvalue decomposition (Ssyevd)
// ---------------------------------------------------------------------------

cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo,
                                             int n,
                                             const float* A, int lda,
                                             const float* W,
                                             int* lwork);

cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo,
                                  int n,
                                  float* A, int lda,
                                  float* W,
                                  float* workspace, int lwork,
                                  int* info);

#ifdef __cplusplus
}
#endif
