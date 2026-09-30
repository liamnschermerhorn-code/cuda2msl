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

static void test_softmax_sums_to_one() {
    printf("  test_softmax_sums_to_one... ");
    const int N = 1, C = 10;
    float h_x[C] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float h_y[C] = {0};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_x, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, C * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t xDesc, yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    // Treat as (N=1, C=10, H=1, W=1) for instance-mode softmax
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha, xDesc, d_x, &beta, yDesc, d_y));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, C * sizeof(float), cudaMemcpyDeviceToHost));

    // All outputs should be positive
    float sum = 0.0f;
    for (int i = 0; i < C; i++) {
        assert(h_y[i] > 0.0f);
        assert(h_y[i] <= 1.0f);
        sum += h_y[i];
    }

    // Sum should be ~1.0
    if (fabs(sum - 1.0f) > 1e-4) {
        fprintf(stderr, "FAIL: softmax sum = %f (expected 1.0)\n", sum);
        exit(1);
    }

    // Monotonicity: larger inputs should have larger outputs
    for (int i = 1; i < C; i++) {
        assert(h_y[i] > h_y[i - 1]);
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    printf("PASS\n");
}

static void test_softmax_batch() {
    printf("  test_softmax_batch... ");
    // 2 batch elements, 4 classes each
    const int N = 2, C = 4;
    float h_x[N * C] = {
        1, 2, 3, 4,   // batch 0
        4, 3, 2, 1    // batch 1
    };
    float h_y[N * C] = {0};

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * C * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t xDesc, yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 1, 1));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                    &alpha, xDesc, d_x, &beta, yDesc, d_y));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * C * sizeof(float), cudaMemcpyDeviceToHost));

    // Each batch element should sum to 1
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += h_y[n * C + c];
        }
        if (fabs(sum - 1.0f) > 1e-4) {
            fprintf(stderr, "FAIL: batch %d softmax sum = %f\n", n, sum);
            exit(1);
        }
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroy(handle));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    printf("PASS\n");
}

int main() {
    printf("=== Softmax Tests ===\n");
    test_softmax_sums_to_one();
    test_softmax_batch();
    printf("All softmax tests PASSED\n");
    return 0;
}
