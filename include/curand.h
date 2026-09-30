#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's curand.h

#include "cuda_runtime_api.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// cuRAND status codes
typedef enum {
    CURAND_STATUS_SUCCESS                   = 0,
    CURAND_STATUS_VERSION_MISMATCH          = 100,
    CURAND_STATUS_NOT_INITIALIZED           = 101,
    CURAND_STATUS_ALLOCATION_FAILED         = 102,
    CURAND_STATUS_TYPE_ERROR                = 103,
    CURAND_STATUS_OUT_OF_RANGE              = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE       = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE            = 201,
    CURAND_STATUS_PREEXISTING_FAILURE       = 202,
    CURAND_STATUS_INITIALIZATION_FAILED     = 203,
    CURAND_STATUS_ARCH_MISMATCH             = 204,
    CURAND_STATUS_INVALID_VALUE             = 400,
    CURAND_STATUS_INTERNAL_ERROR            = 999
} curandStatus_t;

// cuRAND RNG types
typedef enum {
    CURAND_RNG_TEST                    = 0,
    CURAND_RNG_PSEUDO_DEFAULT          = 100,
    CURAND_RNG_PSEUDO_XORWOW           = 101,
    CURAND_RNG_PSEUDO_MRG32K3A         = 121,
    CURAND_RNG_PSEUDO_MTGP32           = 141,
    CURAND_RNG_PSEUDO_MT19937          = 142,
    CURAND_RNG_PSEUDO_PHILOX4_32_10    = 161,
    CURAND_RNG_QUASI_DEFAULT           = 200,
    CURAND_RNG_QUASI_SOBOL32           = 201,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
    CURAND_RNG_QUASI_SOBOL64           = 203,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
} curandRngType_t;

// Opaque generator handle
typedef struct curandGenerator_st* curandGenerator_t;

// ---------------------------------------------------------------------------
// Generator lifecycle
// ---------------------------------------------------------------------------

curandStatus_t curandCreateGenerator(curandGenerator_t* generator,
                                     curandRngType_t rng_type);

curandStatus_t curandDestroyGenerator(curandGenerator_t generator);

// ---------------------------------------------------------------------------
// Generator configuration
// ---------------------------------------------------------------------------

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                  unsigned long long seed);

curandStatus_t curandSetStream(curandGenerator_t generator,
                               cudaStream_t stream);

// ---------------------------------------------------------------------------
// Generation functions
// ---------------------------------------------------------------------------

// Generate 32-bit unsigned integers
curandStatus_t curandGenerate(curandGenerator_t generator,
                              unsigned int* outputPtr,
                              size_t num);

// Generate uniformly distributed floats in (0, 1]
curandStatus_t curandGenerateUniform(curandGenerator_t generator,
                                     float* outputPtr,
                                     size_t num);

// Generate uniformly distributed doubles in (0, 1]
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator,
                                           double* outputPtr,
                                           size_t num);

// Generate normally distributed floats with given mean and stddev
curandStatus_t curandGenerateNormal(curandGenerator_t generator,
                                    float* outputPtr,
                                    size_t n,
                                    float mean,
                                    float stddev);

// Generate normally distributed doubles with given mean and stddev
curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator,
                                          double* outputPtr,
                                          size_t n,
                                          double mean,
                                          double stddev);

#ifdef __cplusplus
}
#endif
