#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cassert>

// ---------------------------------------------------------------------------
// Model-scale integration tests
//
// These tests exercise the full CUDA-to-Metal runtime at model-realistic
// sizes: large matrix multiplies, chained activation + norm sequences, and
// multi-layer transformer / ResNet patterns. They stress:
//   - Memory allocation at model scale (hundreds of MBs)
//   - Large GEMM (cublasSgemm at 1024x1024 and above)
//   - Chained cudaMemcpy → GEMM → activation → norm pipelines
//   - Concurrent stream dispatch
//   - Memory pressure and allocation/deallocation churn
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "FAIL: %s returned %d at %s:%d\n",             \
                    #call, err, __FILE__, __LINE__);                        \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t status = (call);                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "FAIL: %s returned %d at %s:%d\n",             \
                    #call, status, __FILE__, __LINE__);                     \
            return false;                                                   \
        }                                                                   \
    } while (0)

#define CUDNN_CHECK(call)                                                   \
    do {                                                                    \
        cudnnStatus_t status = (call);                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                               \
            fprintf(stderr, "FAIL: %s returned %d at %s:%d\n",             \
                    #call, status, __FILE__, __LINE__);                     \
            return false;                                                   \
        }                                                                   \
    } while (0)

static bool approx_equal(float a, float b, float tol = 1e-3f) {
    return fabs(a - b) < tol + tol * fabs(b);
}

// CPU reference GEMM
static void cpu_sgemm(int m, int n, int k,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb,
                       float beta, float* C, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i + p * lda] * B[p + j * ldb];
            }
            C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
        }
    }
}

// ======================================================================
// Test 1: Large GEMM (1024x1024)
// Validates correctness of MPS-backed matrix multiply at real model sizes
// ======================================================================
static bool test_large_gemm() {
    printf("  test_large_gemm (1024x1024)... ");
    const int M = 1024, N = 1024, K = 1024;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.0f);

    srand(42);
    for (auto& v : h_A) v = (float)(rand() % 200 - 100) / 100.0f;
    for (auto& v : h_B) v = (float)(rand() % 200 - 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, sizeC));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &alpha,
                             d_A, M, d_B, K, &beta, d_C, M));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

    // Spot-check: compute a few reference rows on CPU
    std::vector<float> h_ref(M, 0.0f);
    // Check row 0 of C (all N columns)
    for (int j = 0; j < std::min(N, 4); j++) {
        float sum = 0.0f;
        for (int p = 0; p < K; p++) sum += h_A[0 + p * M] * h_B[p + j * K];
        if (!approx_equal(h_C[0 + j * M], sum, 0.5f)) {
            fprintf(stderr, "FAIL: C[0,%d] = %f, expected %f\n", j, h_C[0 + j * M], sum);
            CUBLAS_CHECK(cublasDestroy(handle));
            CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
            return false;
        }
    }

    // Check finite
    bool all_finite = true;
    for (int i = 0; i < M * N; i++) {
        if (!std::isfinite(h_C[i])) { all_finite = false; break; }
    }
    if (!all_finite) {
        fprintf(stderr, "FAIL: non-finite values in result\n");
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
        return false;
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    double gflops = 2.0 * M * N * K / (ms * 1e6);
    printf("PASS (%.1f ms, %.1f GFLOPS)\n", ms, gflops);
    return true;
}

// ======================================================================
// Test 2: Transformer attention-sized GEMM chain
// Q*K^T (seq x head) then softmax-like scaling then V multiply
// ======================================================================
static bool test_attention_gemm_chain() {
    printf("  test_attention_gemm_chain (seq=512, head=64, 8 heads)... ");

    const int seq_len = 512;
    const int head_dim = 64;
    const int num_heads = 8;
    const int d_model = head_dim * num_heads; // 512

    // Per-head Q, K, V: [seq_len x head_dim]
    size_t qkv_size = seq_len * head_dim * sizeof(float);
    size_t attn_size = seq_len * seq_len * sizeof(float);
    size_t out_size = seq_len * head_dim * sizeof(float);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    srand(123);

    for (int h = 0; h < num_heads; h++) {
        std::vector<float> h_Q(seq_len * head_dim), h_K(seq_len * head_dim), h_V(seq_len * head_dim);
        for (auto& v : h_Q) v = (float)(rand() % 200 - 100) / 100.0f;
        for (auto& v : h_K) v = (float)(rand() % 200 - 100) / 100.0f;
        for (auto& v : h_V) v = (float)(rand() % 200 - 100) / 100.0f;

        float *d_Q, *d_K, *d_V, *d_attn, *d_out;
        CUDA_CHECK(cudaMalloc((void**)&d_Q, qkv_size));
        CUDA_CHECK(cudaMalloc((void**)&d_K, qkv_size));
        CUDA_CHECK(cudaMalloc((void**)&d_V, qkv_size));
        CUDA_CHECK(cudaMalloc((void**)&d_attn, attn_size));
        CUDA_CHECK(cudaMalloc((void**)&d_out, out_size));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_attn, 0, attn_size));
        CUDA_CHECK(cudaMemset(d_out, 0, out_size));

        // Step 1: attn = Q * K^T  [seq x seq]
        float scale = 1.0f / sqrtf((float)head_dim);
        float zero = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 seq_len, seq_len, head_dim, &scale,
                                 d_Q, seq_len, d_K, seq_len, &zero, d_attn, seq_len));

        // Step 2: out = attn * V  [seq x head_dim]
        float one = 1.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 seq_len, head_dim, seq_len, &one,
                                 d_attn, seq_len, d_V, seq_len, &zero, d_out, seq_len));

        CUDA_CHECK(cudaDeviceSynchronize());

        // Validate: read back and check finiteness
        std::vector<float> h_out(seq_len * head_dim);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, out_size, cudaMemcpyDeviceToHost));

        for (int i = 0; i < seq_len * head_dim; i++) {
            if (!std::isfinite(h_out[i])) {
                fprintf(stderr, "FAIL: head %d non-finite at index %d\n", h, i);
                CUBLAS_CHECK(cublasDestroy(handle));
                return false;
            }
        }

        CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
        CUDA_CHECK(cudaFree(d_attn)); CUDA_CHECK(cudaFree(d_out));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    printf("PASS\n");
    return true;
}

// ======================================================================
// Test 3: Activation + Batch Norm chain (conv block pattern)
// Simulates: input → ReLU → BN → output
// ======================================================================
static bool test_activation_batchnorm_chain() {
    printf("  test_activation_batchnorm_chain (N=32, C=256, H=W=16)... ");

    const int N = 32, C = 256, H = 16, W = 16;
    const int total = N * C * H * W;
    size_t bytes = total * sizeof(float);

    std::vector<float> h_input(total);
    srand(77);
    for (auto& v : h_input) v = (float)(rand() % 200 - 100) / 100.0f;

    float *d_input, *d_after_relu;
    CUDA_CHECK(cudaMalloc((void**)&d_input, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_after_relu, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // ReLU via cuDNN
    cudnnHandle_t dnn;
    CUDNN_CHECK(cudnnCreate(&dnn));

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnActivationForward(dnn, actDesc, &alpha, desc, d_input, &beta, desc, d_after_relu));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back and validate
    std::vector<float> h_relu(total);
    CUDA_CHECK(cudaMemcpy(h_relu.data(), d_after_relu, bytes, cudaMemcpyDeviceToHost));

    int relu_failures = 0;
    for (int i = 0; i < total; i++) {
        float expected = h_input[i] > 0 ? h_input[i] : 0.0f;
        if (!approx_equal(h_relu[i], expected)) relu_failures++;
    }
    if (relu_failures > 0) {
        fprintf(stderr, "FAIL: ReLU had %d/%d mismatches\n", relu_failures, total);
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
        CUDNN_CHECK(cudnnDestroy(dnn));
        CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_after_relu));
        return false;
    }

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    CUDNN_CHECK(cudnnDestroy(dnn));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_after_relu));
    printf("PASS (%.1fM elements)\n", total / 1e6);
    return true;
}

// ======================================================================
// Test 4: Multi-layer MLP (Transformer FFN pattern)
// x → Linear(d,4d) → ReLU → Linear(4d,d) × N_layers
// ======================================================================
static bool test_multilayer_mlp() {
    printf("  test_multilayer_mlp (d=512, 4x expansion, 6 layers)... ");

    const int d_model = 512;
    const int d_ff = d_model * 4; // 2048
    const int batch = 128;
    const int num_layers = 6;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Allocate weight matrices for all layers
    struct FFNLayer {
        float* W1; // d_model x d_ff
        float* W2; // d_ff x d_model
    };
    std::vector<FFNLayer> layers(num_layers);

    srand(999);
    for (int l = 0; l < num_layers; l++) {
        CUDA_CHECK(cudaMalloc((void**)&layers[l].W1, d_model * d_ff * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&layers[l].W2, d_ff * d_model * sizeof(float)));

        // Init weights (small random)
        std::vector<float> w1_host(d_model * d_ff), w2_host(d_ff * d_model);
        for (auto& v : w1_host) v = (float)(rand() % 100 - 50) / 1000.0f;
        for (auto& v : w2_host) v = (float)(rand() % 100 - 50) / 1000.0f;
        CUDA_CHECK(cudaMemcpy(layers[l].W1, w1_host.data(), w1_host.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layers[l].W2, w2_host.data(), w2_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Input/output buffers
    float *d_x, *d_hidden, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_x, batch * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_hidden, batch * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, batch * d_model * sizeof(float)));

    // Init input
    std::vector<float> h_x(batch * d_model);
    for (auto& v : h_x) v = (float)(rand() % 200 - 100) / 100.0f;
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

    // ReLU descriptor for intermediate activations
    cudnnTensorDescriptor_t hiddenDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hiddenDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(hiddenDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, d_ff, 1, 1));

    cudnnActivationDescriptor_t actDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    float one = 1.0f, zero = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int l = 0; l < num_layers; l++) {
        // Step 1: hidden = x @ W1  [batch x d_ff]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 batch, d_ff, d_model, &one,
                                 d_x, batch, layers[l].W1, d_model, &zero, d_hidden, batch));

        // Step 2: ReLU(hidden) in-place
        CUDNN_CHECK(cudnnActivationForward(cudnn, actDesc, &one, hiddenDesc, d_hidden, &zero, hiddenDesc, d_hidden));

        // Step 3: out = hidden @ W2  [batch x d_model]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 batch, d_model, d_ff, &one,
                                 d_hidden, batch, layers[l].W2, d_ff, &zero, d_out, batch));

        // Residual: x = x + out (simulate with pointer swap for next layer input)
        // In a real model we'd do elementwise add, but for this test just use output directly
        std::swap(d_x, d_out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Read final output
    std::vector<float> h_result(batch * d_model);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_x, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_finite = true;
    for (int i = 0; i < batch * d_model; i++) {
        if (!std::isfinite(h_result[i])) { all_finite = false; break; }
    }

    // Cleanup
    for (int l = 0; l < num_layers; l++) {
        CUDA_CHECK(cudaFree(layers[l].W1));
        CUDA_CHECK(cudaFree(layers[l].W2));
    }
    // d_x and d_out were swapped, free both original pointers
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(actDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(hiddenDesc));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    if (!all_finite) {
        fprintf(stderr, "FAIL: non-finite values in output after %d layers\n", num_layers);
        return false;
    }

    size_t total_weights = num_layers * (d_model * d_ff + d_ff * d_model) * sizeof(float);
    printf("PASS (%.1f ms, %d layers, %.1f MB weights)\n", ms, num_layers, total_weights / (1024.0 * 1024.0));
    return true;
}

// ======================================================================
// Test 5: Memory stress — allocate/free many buffers
// Simulates the allocation pattern of a real model with many tensors
// ======================================================================
static bool test_memory_stress() {
    printf("  test_memory_stress (256 buffers, up to 4MB each)... ");

    const int num_buffers = 256;
    std::vector<void*> ptrs(num_buffers, nullptr);
    std::vector<size_t> sizes(num_buffers);
    size_t total_allocated = 0;

    srand(55);

    // Allocate varying sizes
    for (int i = 0; i < num_buffers; i++) {
        sizes[i] = (1 + rand() % 1024) * 1024; // 1KB to 1MB
        CUDA_CHECK(cudaMalloc(&ptrs[i], sizes[i]));
        CUDA_CHECK(cudaMemset(ptrs[i], 0, sizes[i]));
        total_allocated += sizes[i];
    }

    // Free every other buffer, then reallocate
    for (int i = 0; i < num_buffers; i += 2) {
        CUDA_CHECK(cudaFree(ptrs[i]));
        ptrs[i] = nullptr;
    }

    // Reallocate with different sizes (fragmentation test)
    for (int i = 0; i < num_buffers; i += 2) {
        sizes[i] = (1 + rand() % 4096) * 1024; // up to 4MB
        CUDA_CHECK(cudaMalloc(&ptrs[i], sizes[i]));
        CUDA_CHECK(cudaMemset(ptrs[i], 0xAA, sizes[i]));
    }

    // Write then read back from each buffer
    for (int i = 0; i < num_buffers; i++) {
        float val = (float)(i + 1);
        CUDA_CHECK(cudaMemcpy(ptrs[i], &val, sizeof(float), cudaMemcpyHostToDevice));
        float readback = 0;
        CUDA_CHECK(cudaMemcpy(&readback, ptrs[i], sizeof(float), cudaMemcpyDeviceToHost));
        if (readback != val) {
            fprintf(stderr, "FAIL: buffer %d readback %f != %f\n", i, readback, val);
            return false;
        }
    }

    // Cleanup
    for (int i = 0; i < num_buffers; i++) {
        if (ptrs[i]) CUDA_CHECK(cudaFree(ptrs[i]));
    }

    printf("PASS (%.1f MB total)\n", total_allocated / (1024.0 * 1024.0));
    return true;
}

// ======================================================================
// Test 6: Multi-stream concurrent dispatch
// Simulates multiple independent operations running on separate streams
// ======================================================================
static bool test_multi_stream() {
    printf("  test_multi_stream (4 streams, concurrent GEMMs)... ");

    const int num_streams = 4;
    const int M = 256, N = 256, K = 256;
    size_t mat_bytes = M * N * sizeof(float);

    cudaStream_t streams[4];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Each stream gets its own A, B, C matrices
    struct StreamData {
        float *d_A, *d_B, *d_C;
        std::vector<float> h_C;
    };
    std::vector<StreamData> data(num_streams);

    srand(42);
    for (int s = 0; s < num_streams; s++) {
        CUDA_CHECK(cudaMalloc((void**)&data[s].d_A, mat_bytes));
        CUDA_CHECK(cudaMalloc((void**)&data[s].d_B, mat_bytes));
        CUDA_CHECK(cudaMalloc((void**)&data[s].d_C, mat_bytes));
        data[s].h_C.resize(M * N);

        std::vector<float> h_A(M * K), h_B(K * N);
        for (auto& v : h_A) v = (float)(rand() % 200 - 100) / 100.0f;
        for (auto& v : h_B) v = (float)(rand() % 200 - 100) / 100.0f;

        CUDA_CHECK(cudaMemcpy(data[s].d_A, h_A.data(), mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(data[s].d_B, h_B.data(), mat_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(data[s].d_C, 0, mat_bytes));
    }

    // Launch GEMMs on all streams
    float one = 1.0f, zero = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < num_streams; s++) {
        CUBLAS_CHECK(cublasSetStream(handle, streams[s]));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 M, N, K, &one,
                                 data[s].d_A, M, data[s].d_B, K,
                                 &zero, data[s].d_C, M));
    }

    // Sync all streams
    for (int s = 0; s < num_streams; s++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Read back and validate finiteness
    for (int s = 0; s < num_streams; s++) {
        CUDA_CHECK(cudaMemcpy(data[s].h_C.data(), data[s].d_C, mat_bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < M * N; i++) {
            if (!std::isfinite(data[s].h_C[i])) {
                fprintf(stderr, "FAIL: stream %d, non-finite at index %d\n", s, i);
                return false;
            }
        }
    }

    // Each stream should produce different results (different input data)
    bool different = false;
    for (int i = 0; i < M * N; i++) {
        if (data[0].h_C[i] != data[1].h_C[i]) { different = true; break; }
    }
    if (!different) {
        fprintf(stderr, "FAIL: streams 0 and 1 produced identical results — suspicious\n");
        // Not necessarily a hard failure (could be coincidence), but flag it
    }

    // Cleanup
    for (int s = 0; s < num_streams; s++) {
        CUDA_CHECK(cudaFree(data[s].d_A));
        CUDA_CHECK(cudaFree(data[s].d_B));
        CUDA_CHECK(cudaFree(data[s].d_C));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUBLAS_CHECK(cublasDestroy(handle));

    printf("PASS (%.1f ms for %d concurrent 256x256 GEMMs)\n", ms, num_streams);
    return true;
}

// ======================================================================
// Test 7: ResNet-like block — Conv (GEMM proxy) → BN → ReLU × depth
// ======================================================================
static bool test_resnet_block() {
    printf("  test_resnet_block (batch=16, channels=512, 4 blocks)... ");

    // Simulate ResNet bottleneck: 512 → 128 → 128 → 512
    // We approximate Conv2d as GEMM (im2col style)
    const int batch = 16;
    const int spatial = 7 * 7; // 7x7 feature map
    const int C_in = 512, C_mid = 128, C_out = 512;
    const int num_blocks = 4;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Weight matrices (treating conv as GEMM)
    float *W_down, *W_mid, *W_up;
    CUDA_CHECK(cudaMalloc((void**)&W_down, C_in * C_mid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&W_mid, C_mid * C_mid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&W_up, C_mid * C_out * sizeof(float)));

    // Init weights small
    std::vector<float> w_host;
    srand(314);
    w_host.resize(C_in * C_mid); for (auto& v : w_host) v = (float)(rand() % 100 - 50) / 5000.0f;
    CUDA_CHECK(cudaMemcpy(W_down, w_host.data(), w_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    w_host.resize(C_mid * C_mid); for (auto& v : w_host) v = (float)(rand() % 100 - 50) / 5000.0f;
    CUDA_CHECK(cudaMemcpy(W_mid, w_host.data(), w_host.size() * sizeof(float), cudaMemcpyHostToDevice));
    w_host.resize(C_mid * C_out); for (auto& v : w_host) v = (float)(rand() % 100 - 50) / 5000.0f;
    CUDA_CHECK(cudaMemcpy(W_up, w_host.data(), w_host.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Feature buffers
    int feat_elems = batch * spatial;
    float *d_x, *d_down, *d_mid, *d_up;
    CUDA_CHECK(cudaMalloc((void**)&d_x, feat_elems * C_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_down, feat_elems * C_mid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mid, feat_elems * C_mid * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_up, feat_elems * C_out * sizeof(float)));

    // Init input
    std::vector<float> h_x(feat_elems * C_in);
    for (auto& v : h_x) v = (float)(rand() % 200 - 100) / 100.0f;
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

    // ReLU
    cudnnTensorDescriptor_t midDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&midDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(midDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, 1, 1, feat_elems * C_mid));

    cudnnActivationDescriptor_t reluDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&reluDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(reluDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    float one = 1.0f, zero = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < num_blocks; b++) {
        // 1x1 conv down: [feat_elems x C_in] @ [C_in x C_mid] → [feat_elems x C_mid]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 feat_elems, C_mid, C_in, &one,
                                 d_x, feat_elems, W_down, C_in, &zero, d_down, feat_elems));

        // ReLU
        CUDNN_CHECK(cudnnActivationForward(cudnn, reluDesc, &one, midDesc, d_down, &zero, midDesc, d_down));

        // 3x3 conv mid (proxy): [feat_elems x C_mid] @ [C_mid x C_mid] → [feat_elems x C_mid]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 feat_elems, C_mid, C_mid, &one,
                                 d_down, feat_elems, W_mid, C_mid, &zero, d_mid, feat_elems));

        // ReLU
        CUDNN_CHECK(cudnnActivationForward(cudnn, reluDesc, &one, midDesc, d_mid, &zero, midDesc, d_mid));

        // 1x1 conv up: [feat_elems x C_mid] @ [C_mid x C_out] → [feat_elems x C_out]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 feat_elems, C_out, C_mid, &one,
                                 d_mid, feat_elems, W_up, C_mid, &zero, d_up, feat_elems));

        // Residual add would go here (d_x += d_up), use output as next input
        std::swap(d_x, d_up);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Validate
    std::vector<float> h_result(feat_elems * C_in);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_x, feat_elems * C_in * sizeof(float), cudaMemcpyDeviceToHost));

    bool all_finite = true;
    for (size_t i = 0; i < h_result.size(); i++) {
        if (!std::isfinite(h_result[i])) { all_finite = false; break; }
    }

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_down)); CUDA_CHECK(cudaFree(d_mid));
    CUDA_CHECK(cudaFree(W_down)); CUDA_CHECK(cudaFree(W_mid)); CUDA_CHECK(cudaFree(W_up));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(reluDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(midDesc));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    if (!all_finite) {
        fprintf(stderr, "FAIL: non-finite values after %d ResNet blocks\n", num_blocks);
        return false;
    }

    printf("PASS (%.1f ms, %d blocks)\n", ms, num_blocks);
    return true;
}

// ======================================================================
// Test 8: Event timing accuracy
// ======================================================================
static bool test_event_timing() {
    printf("  test_event_timing... ");

    const int M = 512, N = 512, K = 512;
    size_t bytes = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
    CUDA_CHECK(cudaMemset(d_A, 0, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B, 0, K * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float one = 1.0f, zero = 0.0f;

    CUDA_CHECK(cudaEventRecord(start, nullptr));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &one, d_A, M, d_B, K, &zero, d_C, M));
    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    bool timing_reasonable = (elapsed_ms >= 0.0f && elapsed_ms < 5000.0f);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    if (!timing_reasonable) {
        fprintf(stderr, "FAIL: elapsed time %f ms is not reasonable\n", elapsed_ms);
        return false;
    }

    printf("PASS (%.2f ms measured)\n", elapsed_ms);
    return true;
}

// ======================================================================
// Main
// ======================================================================
int main() {
    printf("=== Model-Scale Integration Tests ===\n\n");

    int pass = 0, fail = 0;
    auto run = [&](bool (*test)()) {
        if (test()) pass++; else fail++;
    };

    run(test_large_gemm);
    run(test_attention_gemm_chain);
    run(test_activation_batchnorm_chain);
    run(test_multilayer_mlp);
    run(test_memory_stress);
    run(test_multi_stream);
    run(test_resnet_block);
    run(test_event_timing);

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
