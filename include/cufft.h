#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's cufft.h

#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// cuFFT result codes
// ---------------------------------------------------------------------------

typedef enum {
    CUFFT_SUCCESS                   = 0x0,
    CUFFT_INVALID_PLAN              = 0x1,
    CUFFT_ALLOC_FAILED              = 0x2,
    CUFFT_INVALID_TYPE              = 0x3,
    CUFFT_INVALID_VALUE             = 0x4,
    CUFFT_INTERNAL_ERROR            = 0x5,
    CUFFT_EXEC_FAILED               = 0x6,
    CUFFT_SETUP_FAILED              = 0x7,
    CUFFT_INVALID_SIZE              = 0x8,
    CUFFT_UNALIGNED_DATA            = 0x9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
    CUFFT_INVALID_DEVICE            = 0xB,
    CUFFT_PARSE_ERROR               = 0xC,
    CUFFT_NO_WORKSPACE              = 0xD,
    CUFFT_NOT_IMPLEMENTED           = 0xE,
    CUFFT_NOT_SUPPORTED             = 0xF
} cufftResult;

// ---------------------------------------------------------------------------
// cuFFT transform types
// ---------------------------------------------------------------------------

typedef enum {
    CUFFT_R2C = 0x2a,  // Real to complex (interleaved)
    CUFFT_C2R = 0x2c,  // Complex (interleaved) to real
    CUFFT_C2C = 0x29,  // Complex to complex (single precision)
    CUFFT_D2Z = 0x6a,  // Double to double-complex
    CUFFT_Z2D = 0x6c,  // Double-complex to double
    CUFFT_Z2Z = 0x69   // Double-complex to double-complex
} cufftType;

// ---------------------------------------------------------------------------
// Transform direction
// ---------------------------------------------------------------------------

#define CUFFT_FORWARD -1
#define CUFFT_INVERSE  1

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

typedef float  cufftReal;
typedef double cufftDoubleReal;

typedef struct cufftComplex {
    float x;
    float y;
} cufftComplex;

typedef struct cufftDoubleComplex {
    double x;
    double y;
} cufftDoubleComplex;

// ---------------------------------------------------------------------------
// Plan handle
// ---------------------------------------------------------------------------

typedef int cufftHandle;

// ---------------------------------------------------------------------------
// Plan management
// ---------------------------------------------------------------------------

cufftResult cufftPlan1d(cufftHandle* plan,
                        int nx,
                        cufftType type,
                        int batch);

cufftResult cufftPlan2d(cufftHandle* plan,
                        int nx, int ny,
                        cufftType type);

cufftResult cufftPlanMany(cufftHandle* plan,
                          int rank,
                          int* n,
                          int* inembed, int istride, int idist,
                          int* onembed, int ostride, int odist,
                          cufftType type,
                          int batch);

cufftResult cufftDestroy(cufftHandle plan);

// ---------------------------------------------------------------------------
// Stream association
// ---------------------------------------------------------------------------

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Execution — single precision
// ---------------------------------------------------------------------------

cufftResult cufftExecC2C(cufftHandle plan,
                         cufftComplex* idata,
                         cufftComplex* odata,
                         int direction);

cufftResult cufftExecR2C(cufftHandle plan,
                         cufftReal* idata,
                         cufftComplex* odata);

cufftResult cufftExecC2R(cufftHandle plan,
                         cufftComplex* idata,
                         cufftReal* odata);

// ---------------------------------------------------------------------------
// Execution — double precision
// ---------------------------------------------------------------------------

cufftResult cufftExecZ2Z(cufftHandle plan,
                         cufftDoubleComplex* idata,
                         cufftDoubleComplex* odata,
                         int direction);

cufftResult cufftExecD2Z(cufftHandle plan,
                         cufftDoubleReal* idata,
                         cufftDoubleComplex* odata);

cufftResult cufftExecZ2D(cufftHandle plan,
                         cufftDoubleComplex* idata,
                         cufftDoubleReal* odata);

#ifdef __cplusplus
}
#endif
