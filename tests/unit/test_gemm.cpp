#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "FAIL: %s returned %d at %s:%d\n",             \
                    #call, err, __FILE__, __LINE__);                        \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = (call);                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "FAIL: %s returned %d at %s:%d\n",             \
                    #call, status, __FILE__, __LINE__);                     \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static bool approx_equal(float a, float b, float tol = 1e-3f) {
    return fabs(a - b) < tol + tol * fabs(b);
}

// CPU reference GEMM: C = alpha * op(A) * op(B) + beta * C (column-major)
static void cpu_sgemm(cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k,
                      float alpha,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float beta,
                      float* C, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                float a_val = (transa == CUBLAS_OP_N)
                    ? A[i + p * lda]
                    : A[p + i * lda];
                float b_val = (transb == CUBLAS_OP_N)
                    ? B[p + j * ldb]
                    : B[j + p * ldb];
                sum += a_val * b_val;
            }
            C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
        }
    }
}

static void test_identity_gemm() {
    printf("  test_identity_gemm... ");
    // C = I * A, where I is identity matrix
    const int N = 4;
    float h_I[N * N] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float h_A[N * N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float h_C[N * N] = {0};

    float *d_I, *d_A, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_I, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_I, h_I, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, N * N * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N, &alpha,
                             d_I, N, d_A, N, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // C should equal A
    for (int i = 0; i < N * N; i++) {
        if (!approx_equal(h_C[i], h_A[i])) {
            fprintf(stderr, "FAIL at index %d: got %f, expected %f\n",
                    i, h_C[i], h_A[i]);
            exit(1);
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_I));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    printf("PASS\n");
}

static void test_known_answer_gemm() {
    printf("  test_known_answer_gemm... ");
    // Small known-answer test
    // A = [[1, 2], [3, 4]] (column-major: [1, 3, 2, 4])
    // B = [[5, 6], [7, 8]] (column-major: [5, 7, 6, 8])
    // C = A * B = [[19, 22], [43, 50]] (column-major: [19, 43, 22, 50])

    float h_A[] = {1, 3, 2, 4};  // 2x2 column-major
    float h_B[] = {5, 7, 6, 8};  // 2x2 column-major
    float h_C[4] = {0};
    float expected[] = {19, 43, 22, 50};

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, 4 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, 4 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, 4 * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             2, 2, 2, &alpha,
                             d_A, 2, d_B, 2, &beta, d_C, 2));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++) {
        if (!approx_equal(h_C[i], expected[i])) {
            fprintf(stderr, "FAIL at index %d: got %f, expected %f\n",
                    i, h_C[i], expected[i]);
            exit(1);
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    printf("PASS\n");
}

static void test_gemm_with_alpha_beta() {
    printf("  test_gemm_with_alpha_beta... ");
    const int M = 3, N = 3, K = 3;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);

    // Fill with known values
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(i + 1);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(i + 1);
    for (int i = 0; i < M * N; i++) h_C[i] = 1.0f;
    for (int i = 0; i < M * N; i++) h_ref[i] = 1.0f;

    float alpha = 2.0f, beta = 3.0f;
    cpu_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha,
              h_A.data(), M, h_B.data(), K, beta, h_ref.data(), M);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &alpha,
                             d_A, M, d_B, K, &beta, d_C, M));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; i++) {
        if (!approx_equal(h_C[i], h_ref[i])) {
            fprintf(stderr, "FAIL at index %d: got %f, expected %f\n",
                    i, h_C[i], h_ref[i]);
            exit(1);
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    printf("PASS\n");
}

static void test_larger_gemm() {
    printf("  test_larger_gemm... ");
    const int M = 64, N = 32, K = 48;

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f), h_ref(M * N, 0.0f);

    srand(42);
    for (auto& v : h_A) v = (float)(rand() % 100) / 50.0f - 1.0f;
    for (auto& v : h_B) v = (float)(rand() % 100) / 50.0f - 1.0f;

    float alpha = 1.0f, beta = 0.0f;
    cpu_sgemm(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, alpha,
              h_A.data(), M, h_B.data(), K, beta, h_ref.data(), M);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &alpha,
                             d_A, M, d_B, K, &beta, d_C, M));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        if (!approx_equal(h_C[i], h_ref[i], 1e-2f)) {
            if (mismatches < 5) {
                fprintf(stderr, "  Mismatch at %d: got %f, expected %f\n",
                        i, h_C[i], h_ref[i]);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "FAIL: %d/%d mismatches\n", mismatches, M * N);
        exit(1);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    printf("PASS\n");
}

int main() {
    printf("=== GEMM Tests ===\n");
    test_identity_gemm();
    test_known_answer_gemm();
    test_gemm_with_alpha_beta();
    test_larger_gemm();
    printf("All GEMM tests PASSED\n");
    return 0;
}
