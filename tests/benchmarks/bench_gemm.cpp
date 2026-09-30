// GEMM Benchmark — measures GFLOPS across matrix sizes
// Tests cublasSgemm and cublasHgemm throughput

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error: %d at %s:%d\n", st, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

struct BenchResult {
    int M, N, K;
    double gflops;
    double time_ms;
};

static BenchResult bench_sgemm(cublasHandle_t handle, int M, int N, int K, int warmup, int iters) {
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&dB, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&dC, sizeC));

    // Initialize with random data
    std::vector<float> hA(M * K), hB(K * N);
    for (auto& v : hA) v = (float)rand() / RAND_MAX;
    for (auto& v : hB) v = (float)rand() / RAND_MAX;
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeC));

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K, &alpha, dA, M, dB, K, &beta, dC, M));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed runs — use wall-clock with sync for accurate GPU timing
    using clock = std::chrono::high_resolution_clock;

    CHECK_CUDA(cudaDeviceSynchronize());
    auto t0 = clock::now();

    for (int i = 0; i < iters; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K, &alpha, dA, M, dB, K, &beta, dC, M));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    auto t1 = clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double avg_ms = elapsed_ms / iters;
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    return {M, N, K, gflops, avg_ms};
}

int main() {
    printf("=== GEMM Benchmark ===\n\n");

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %zu MB\n\n", prop.totalGlobalMem / (1024 * 1024));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Square matrix sizes
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-8s %-8s %-8s  %12s  %12s\n", "M", "N", "K", "GFLOPS", "Time (ms)");
    printf("%-8s %-8s %-8s  %12s  %12s\n", "----", "----", "----", "--------", "---------");

    for (int i = 0; i < num_sizes; i++) {
        int sz = sizes[i];
        int warmup = 3;
        int iters = (sz <= 512) ? 20 : (sz <= 1024) ? 10 : 5;

        auto r = bench_sgemm(handle, sz, sz, sz, warmup, iters);
        printf("%-8d %-8d %-8d  %12.2f  %12.3f\n", r.M, r.N, r.K, r.gflops, r.time_ms);
    }

    // Common AI inference shapes
    printf("\n--- AI Inference Shapes (FP32) ---\n");
    printf("%-20s %-8s %-8s %-8s  %12s  %12s\n", "Layer", "M", "N", "K", "GFLOPS", "Time (ms)");
    printf("%-20s %-8s %-8s %-8s  %12s  %12s\n", "-----", "----", "----", "----", "--------", "---------");

    struct { const char* name; int M, N, K; } ai_shapes[] = {
        {"ResNet FC",          1,    1000,  2048},
        {"BERT Attention",     512,  64,    64},
        {"BERT FFN",           512,  3072,  768},
        {"GPT-2 Attention",    1024, 64,    64},
        {"GPT-2 FFN",          1024, 4096,  1024},
        {"Batch GEMM 32",      32,   1024,  1024},
    };

    for (auto& s : ai_shapes) {
        auto r = bench_sgemm(handle, s.M, s.N, s.K, 3, 10);
        printf("%-20s %-8d %-8d %-8d  %12.2f  %12.3f\n", s.name, r.M, r.N, r.K, r.gflops, r.time_ms);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    printf("\nBenchmark complete.\n");
    return 0;
}
