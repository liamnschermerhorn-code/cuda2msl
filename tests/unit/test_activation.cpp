#include <cuda_runtime.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "FAIL: %s returned %d at %s:%d\n",             \
                    #call, err, __FILE__, __LINE__);                        \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CUDNN_CHECK(call)                                                   \
    do {                                                                    \
        cudnnStatus_t status = (call);                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                               \
            fprintf(stderr, "FAIL: %s returned %d at %s:%d\n",             \
                    #call, status, __FILE__, __LINE__);                     \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static void test_relu() {
    printf("  test_relu... ");
    const int N = 8;
    float h_x[] = {-3, -1, -0.5, 0, 0.5, 1, 2, 5};
    float h_y[N] = {0};
    float expected[] = {0, 0, 0, 0, 0.5, 1, 2, 5};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t xDesc, yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(handle, actDesc, &alpha, xDesc, d_x, &beta, yDesc, d_y));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (fabs(h_y[i] - expected[i]) > 1e-5) {
            fprintf(stderr, "FAIL at %d: got %f, expected %f\n", i, h_y[i], expected[i]);
            exit(1);
        }
    }

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    printf("PASS\n");
}

static void test_sigmoid() {
    printf("  test_sigmoid... ");
    const int N = 4;
    float h_x[] = {-2, 0, 1, 5};
    float h_y[N] = {0};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t xDesc, yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, N));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(handle, actDesc, &alpha, xDesc, d_x, &beta, yDesc, d_y));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // All sigmoid outputs should be in [0, 1]
    for (int i = 0; i < N; i++) {
        assert(h_y[i] >= 0.0f && h_y[i] <= 1.0f);
    }
    // sigmoid(0) ~= 0.5
    assert(fabs(h_y[1] - 0.5f) < 1e-5);

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    printf("PASS\n");
}

int main() {
    printf("=== Activation Tests ===\n");
    test_relu();
    test_sigmoid();
    printf("All activation tests PASSED\n");
    return 0;
}
