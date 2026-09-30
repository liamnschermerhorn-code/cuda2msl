#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's cudnn.h (inference subset)

#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// cuDNN status codes
typedef enum {
    CUDNN_STATUS_SUCCESS          = 0,
    CUDNN_STATUS_NOT_INITIALIZED  = 1,
    CUDNN_STATUS_ALLOC_FAILED     = 2,
    CUDNN_STATUS_BAD_PARAM        = 3,
    CUDNN_STATUS_INTERNAL_ERROR   = 4,
    CUDNN_STATUS_INVALID_VALUE    = 5,
    CUDNN_STATUS_ARCH_MISMATCH    = 6,
    CUDNN_STATUS_MAPPING_ERROR    = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED    = 9
} cudnnStatus_t;

// Data types
typedef enum {
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF   = 2
} cudnnDataType_t;

// Tensor format
typedef enum {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1
} cudnnTensorFormat_t;

// Convolution mode
typedef enum {
    CUDNN_CONVOLUTION       = 0,
    CUDNN_CROSS_CORRELATION = 1
} cudnnConvolutionMode_t;

// Convolution forward algorithm
typedef enum {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7
} cudnnConvolutionFwdAlgo_t;

// Activation mode
typedef enum {
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU          = 4,
    CUDNN_ACTIVATION_IDENTITY     = 5
} cudnnActivationMode_t;

// Softmax algorithm
typedef enum {
    CUDNN_SOFTMAX_FAST     = 0,
    CUDNN_SOFTMAX_ACCURATE = 1,
    CUDNN_SOFTMAX_LOG      = 2
} cudnnSoftmaxAlgorithm_t;

// Softmax mode
typedef enum {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,
    CUDNN_SOFTMAX_MODE_CHANNEL  = 1
} cudnnSoftmaxMode_t;

// Pooling mode
typedef enum {
    CUDNN_POOLING_MAX                           = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
} cudnnPoolingMode_t;

// Batch normalization mode
typedef enum {
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL        = 1
} cudnnBatchNormMode_t;

// NaN propagation
typedef enum {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN     = 1
} cudnnNanPropagation_t;

// Opaque descriptor handles
typedef struct cudnnContext*               cudnnHandle_t;
typedef struct cudnnTensorStruct*          cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct*          cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct*     cudnnConvolutionDescriptor_t;
typedef struct cudnnActivationStruct*      cudnnActivationDescriptor_t;
typedef struct cudnnPoolingStruct*         cudnnPoolingDescriptor_t;

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Tensor Descriptor
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* desc);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t desc,
                                         cudnnTensorFormat_t format,
                                         cudnnDataType_t dataType,
                                         int n, int c, int h, int w);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t desc);

// ---------------------------------------------------------------------------
// Filter Descriptor
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* desc);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t desc,
                                         cudnnDataType_t dataType,
                                         cudnnTensorFormat_t format,
                                         int k, int c, int h, int w);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t desc);

// ---------------------------------------------------------------------------
// Convolution Descriptor
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* desc);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t desc,
                                              int padH, int padW,
                                              int strideH, int strideW,
                                              int dilationH, int dilationW,
                                              cudnnConvolutionMode_t mode,
                                              cudnnDataType_t computeType);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t desc);

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t inputDesc,
    const cudnnFilterDescriptor_t filterDesc,
    int* n, int* c, int* h, int* w);

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t* sizeInBytes);

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle,
    const void* alpha,
    const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnFilterDescriptor_t wDesc, const void* w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta,
    const cudnnTensorDescriptor_t yDesc, void* y);

// ---------------------------------------------------------------------------
// Activation Descriptor
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* desc);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t desc,
                                           cudnnActivationMode_t mode,
                                           cudnnNanPropagation_t reluNanOpt,
                                           double coef);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t desc);

cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const void* alpha,
    const cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    const cudnnTensorDescriptor_t yDesc, void* y);

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t handle,
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    const void* alpha,
    const cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    const cudnnTensorDescriptor_t yDesc, void* y);

// ---------------------------------------------------------------------------
// Pooling Descriptor
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* desc);
cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t desc,
                                          cudnnPoolingMode_t mode,
                                          cudnnNanPropagation_t nanProp,
                                          int windowH, int windowW,
                                          int padH, int padW,
                                          int strideH, int strideW);
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t desc);

cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t handle,
    const cudnnPoolingDescriptor_t poolingDesc,
    const void* alpha,
    const cudnnTensorDescriptor_t xDesc, const void* x,
    const void* beta,
    const cudnnTensorDescriptor_t yDesc, void* y);

// ---------------------------------------------------------------------------
// Batch Normalization (Inference)
// ---------------------------------------------------------------------------

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void* alpha,
    const void* beta,
    const cudnnTensorDescriptor_t xDesc, const void* x,
    const cudnnTensorDescriptor_t yDesc, void* y,
    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale,
    const void* bnBias,
    const void* estimatedMean,
    const void* estimatedVariance,
    double epsilon);

#ifdef __cplusplus
}
#endif
