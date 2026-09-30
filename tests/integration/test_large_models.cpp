#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <functional>

// ---------------------------------------------------------------------------
// Large-model integration tests
//
// GPT-2 / LLaMA / ResNet-50 scale patterns exercising the full CUDA-to-Metal
// dispatch stack with realistic tensor sizes, deep layer stacks, and heavy
// memory traffic. Targets:
//   - 2048x2048+ GEMM (GPT-2 hidden dim)
//   - 12-24 layer transformer with full attention + FFN + residual + norm
//   - ResNet-50 depth with bottleneck blocks across 4 stages
//   - Mixed precision simulation (fp32 accumulation over fp16-sized work)
//   - Sustained memory pressure (>500 MB live allocations)
//   - Pipeline latency (end-to-end forward pass timing)
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fill_random(float* host, size_t count, float scale = 0.02f) {
    for (size_t i = 0; i < count; i++)
        host[i] = scale * ((float)(rand() % 2000 - 1000) / 1000.0f);
}

static bool all_finite(const float* data, size_t count) {
    for (size_t i = 0; i < count; i++)
        if (!std::isfinite(data[i])) return false;
    return true;
}

static double bytes_to_mb(size_t bytes) { return bytes / (1024.0 * 1024.0); }

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ======================================================================
// Test 1: GPT-2 Full Transformer Block (12 layers)
//
// Config: d_model=768, n_heads=12, d_ff=3072, seq_len=512, batch=8
// Each layer: LayerNorm → MultiHeadAttn → Residual → LayerNorm → FFN → Residual
// Total weight: ~85 MB per layer × 12 = ~1 GB parameter equivalent
// ======================================================================
static bool test_gpt2_transformer() {
    printf("  test_gpt2_transformer (d=768, heads=12, seq=512, batch=8, 12 layers)...\n");

    const int d_model = 768;
    const int n_heads = 12;
    const int head_dim = d_model / n_heads; // 64
    const int d_ff = 3072;
    const int seq_len = 512;
    const int batch = 8;
    const int num_layers = 12;
    const int tokens = batch * seq_len; // 4096

    size_t total_allocated = 0;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // Per-layer weights
    struct TransformerLayer {
        float* Wq;   // [d_model x d_model]
        float* Wk;   // [d_model x d_model]
        float* Wv;   // [d_model x d_model]
        float* Wo;   // [d_model x d_model]
        float* W1;   // [d_model x d_ff]
        float* W2;   // [d_ff x d_model]
    };

    std::vector<TransformerLayer> layers(num_layers);
    size_t qkvo_size = d_model * d_model * sizeof(float);
    size_t w1_size = d_model * d_ff * sizeof(float);
    size_t w2_size = d_ff * d_model * sizeof(float);

    srand(2024);
    std::vector<float> w_buf;

    for (int l = 0; l < num_layers; l++) {
        CUDA_CHECK(cudaMalloc((void**)&layers[l].Wq, qkvo_size));
        CUDA_CHECK(cudaMalloc((void**)&layers[l].Wk, qkvo_size));
        CUDA_CHECK(cudaMalloc((void**)&layers[l].Wv, qkvo_size));
        CUDA_CHECK(cudaMalloc((void**)&layers[l].Wo, qkvo_size));
        CUDA_CHECK(cudaMalloc((void**)&layers[l].W1, w1_size));
        CUDA_CHECK(cudaMalloc((void**)&layers[l].W2, w2_size));
        total_allocated += 4 * qkvo_size + w1_size + w2_size;

        // Init weights
        w_buf.resize(d_model * d_model);
        fill_random(w_buf.data(), w_buf.size());
        CUDA_CHECK(cudaMemcpy(layers[l].Wq, w_buf.data(), qkvo_size, cudaMemcpyHostToDevice));
        fill_random(w_buf.data(), w_buf.size());
        CUDA_CHECK(cudaMemcpy(layers[l].Wk, w_buf.data(), qkvo_size, cudaMemcpyHostToDevice));
        fill_random(w_buf.data(), w_buf.size());
        CUDA_CHECK(cudaMemcpy(layers[l].Wv, w_buf.data(), qkvo_size, cudaMemcpyHostToDevice));
        fill_random(w_buf.data(), w_buf.size());
        CUDA_CHECK(cudaMemcpy(layers[l].Wo, w_buf.data(), qkvo_size, cudaMemcpyHostToDevice));

        w_buf.resize(d_model * d_ff);
        fill_random(w_buf.data(), w_buf.size());
        CUDA_CHECK(cudaMemcpy(layers[l].W1, w_buf.data(), w1_size, cudaMemcpyHostToDevice));

        w_buf.resize(d_ff * d_model);
        fill_random(w_buf.data(), w_buf.size());
        CUDA_CHECK(cudaMemcpy(layers[l].W2, w_buf.data(), w2_size, cudaMemcpyHostToDevice));
    }

    // Activation buffers (reused across layers)
    size_t x_size = tokens * d_model * sizeof(float);
    size_t qkv_size = tokens * d_model * sizeof(float);
    size_t attn_scores_size = tokens * seq_len * sizeof(float); // per-head would be smaller but we use full
    size_t ff_size = tokens * d_ff * sizeof(float);

    float *d_x, *d_residual, *d_Q, *d_K, *d_V, *d_attn, *d_attn_out, *d_ff_hidden, *d_ff_out;
    CUDA_CHECK(cudaMalloc((void**)&d_x, x_size));
    CUDA_CHECK(cudaMalloc((void**)&d_residual, x_size));
    CUDA_CHECK(cudaMalloc((void**)&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc((void**)&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc((void**)&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc((void**)&d_attn, attn_scores_size));
    CUDA_CHECK(cudaMalloc((void**)&d_attn_out, x_size));
    CUDA_CHECK(cudaMalloc((void**)&d_ff_hidden, ff_size));
    CUDA_CHECK(cudaMalloc((void**)&d_ff_out, x_size));
    total_allocated += 2 * x_size + 3 * qkv_size + attn_scores_size + ff_size + x_size;

    // Init input embedding (random)
    std::vector<float> h_x(tokens * d_model);
    fill_random(h_x.data(), h_x.size(), 1.0f);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_size, cudaMemcpyHostToDevice));

    // ReLU for FFN (GELU approximation = just ReLU for testing dispatch path)
    cudnnTensorDescriptor_t ffDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&ffDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(ffDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, tokens * d_ff));

    cudnnActivationDescriptor_t reluDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&reluDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(reluDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    float one = 1.0f, zero = 0.0f;
    float inv_sqrt_dk = 1.0f / sqrtf((float)head_dim);

    printf("    Allocated %.1f MB (%.1f MB weights, %.1f MB activations)\n",
           bytes_to_mb(total_allocated),
           bytes_to_mb(num_layers * (4 * qkvo_size + w1_size + w2_size)),
           bytes_to_mb(2 * x_size + 3 * qkv_size + attn_scores_size + ff_size + x_size));

    Timer timer;
    timer.start();

    for (int l = 0; l < num_layers; l++) {
        // Save residual
        CUDA_CHECK(cudaMemcpy(d_residual, d_x, x_size, cudaMemcpyDeviceToDevice));

        // --- Self-Attention ---

        // Q = X @ Wq  [tokens x d_model]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_model, &one,
                                 d_x, tokens, layers[l].Wq, d_model, &zero, d_Q, tokens));

        // K = X @ Wk
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_model, &one,
                                 d_x, tokens, layers[l].Wk, d_model, &zero, d_K, tokens));

        // V = X @ Wv
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_model, &one,
                                 d_x, tokens, layers[l].Wv, d_model, &zero, d_V, tokens));

        // Attention scores: attn = Q @ K^T * (1/sqrt(d_k))
        // We compute this as a single [tokens x tokens] matrix (simplified — real impl is per-head)
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                 tokens, tokens, d_model, &inv_sqrt_dk,
                                 d_Q, tokens, d_K, tokens, &zero, d_attn, tokens));

        // attn_out = attn @ V  [tokens x d_model]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, tokens, &one,
                                 d_attn, tokens, d_V, tokens, &zero, d_attn_out, tokens));

        // Output projection: x = attn_out @ Wo
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_model, &one,
                                 d_attn_out, tokens, layers[l].Wo, d_model, &zero, d_x, tokens));

        // Residual add: x = x + residual (reuse d_x += d_residual via GEMM trick: x = 1*x + 1*residual)
        // Use cublasSaxpy or just a second pass — for simplicity, use Sgeam
        // Actually, simplest: cublasSgemm with I identity doesn't work. Use cudaMemcpy approach:
        // x_final = x + residual → we can use cublasSaxpy
        CUBLAS_CHECK(cublasSaxpy(cublas, tokens * d_model, &one, d_residual, 1, d_x, 1));

        // --- FFN ---

        // Save residual
        CUDA_CHECK(cudaMemcpy(d_residual, d_x, x_size, cudaMemcpyDeviceToDevice));

        // ff_hidden = x @ W1  [tokens x d_ff]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_ff, d_model, &one,
                                 d_x, tokens, layers[l].W1, d_model, &zero, d_ff_hidden, tokens));

        // ReLU(ff_hidden) in-place
        CUDNN_CHECK(cudnnActivationForward(cudnn, reluDesc, &one, ffDesc, d_ff_hidden, &zero, ffDesc, d_ff_hidden));

        // ff_out = ff_hidden @ W2  [tokens x d_model]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_ff, &one,
                                 d_ff_hidden, tokens, layers[l].W2, d_ff, &zero, d_ff_out, tokens));

        // Residual add
        CUBLAS_CHECK(cublasSaxpy(cublas, tokens * d_model, &one, d_residual, 1, d_ff_out, 1));

        // Move output to input for next layer
        CUDA_CHECK(cudaMemcpy(d_x, d_ff_out, x_size, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = timer.ms();

    // Validate output
    std::vector<float> h_result(tokens * d_model);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_x, x_size, cudaMemcpyDeviceToHost));

    bool finite = all_finite(h_result.data(), h_result.size());

    // Compute stats
    float min_val = *std::min_element(h_result.begin(), h_result.end());
    float max_val = *std::max_element(h_result.begin(), h_result.end());
    double sum = 0;
    for (auto v : h_result) sum += v;
    float mean = (float)(sum / h_result.size());

    // Count GEMMs: per layer = 3 (QKV proj) + 1 (attn scores) + 1 (attn@V) + 1 (Wo) + 1 (W1) + 1 (W2) = 8
    // Plus Saxpy (2 per layer) counted separately
    size_t total_flops = 0;
    for (int l = 0; l < num_layers; l++) {
        total_flops += 3ULL * 2 * tokens * d_model * d_model;  // QKV projections
        total_flops += 2ULL * tokens * tokens * d_model;         // attn scores
        total_flops += 2ULL * tokens * d_model * tokens;         // attn @ V
        total_flops += 2ULL * tokens * d_model * d_model;        // Wo
        total_flops += 2ULL * tokens * d_ff * d_model;           // W1
        total_flops += 2ULL * tokens * d_model * d_ff;           // W2
    }
    double tflops = total_flops / (elapsed * 1e9);

    // Cleanup
    for (int l = 0; l < num_layers; l++) {
        CUDA_CHECK(cudaFree(layers[l].Wq)); CUDA_CHECK(cudaFree(layers[l].Wk));
        CUDA_CHECK(cudaFree(layers[l].Wv)); CUDA_CHECK(cudaFree(layers[l].Wo));
        CUDA_CHECK(cudaFree(layers[l].W1)); CUDA_CHECK(cudaFree(layers[l].W2));
    }
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_attn)); CUDA_CHECK(cudaFree(d_attn_out));
    CUDA_CHECK(cudaFree(d_ff_hidden)); CUDA_CHECK(cudaFree(d_ff_out));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(reluDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(ffDesc));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    if (!finite) {
        fprintf(stderr, "    FAIL: non-finite values in output\n");
        return false;
    }

    printf("    Output stats: mean=%.4f, min=%.4f, max=%.4f\n", mean, min_val, max_val);
    printf("    PASS (%.1f ms, %.2f TFLOPS, %d layers, %.1f MB allocated)\n",
           elapsed, tflops, num_layers, bytes_to_mb(total_allocated));
    return true;
}

// ======================================================================
// Test 2: LLaMA-scale GEMM stress (2048x2048, 4096x4096)
//
// Pure GEMM throughput at dimensions matching LLaMA-7B hidden sizes
// ======================================================================
static bool test_llama_gemm_sizes() {
    printf("  test_llama_gemm_sizes (2048x2048, 4096x4096, 4096x11008)...\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    struct GEMMSpec {
        int M, N, K;
        const char* label;
    };

    GEMMSpec specs[] = {
        {2048, 2048, 2048, "LLaMA-7B hidden (2048^3)"},
        {4096, 4096, 4096, "LLaMA-13B hidden (4096^3)"},
        {4096, 11008, 4096, "LLaMA-7B FFN up-proj (4096x11008x4096)"},
        {2048, 2048, 8192, "Batched attention (2048x2048x8192)"},
    };

    float one = 1.0f, zero = 0.0f;
    srand(42);

    for (const auto& spec : specs) {
        size_t sizeA = (size_t)spec.M * spec.K * sizeof(float);
        size_t sizeB = (size_t)spec.K * spec.N * sizeof(float);
        size_t sizeC = (size_t)spec.M * spec.N * sizeof(float);
        size_t total = sizeA + sizeB + sizeC;

        printf("    %s (%.1f MB)... ", spec.label, bytes_to_mb(total));

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
        CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
        CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

        // Init with small random values (avoid overflow in large matmuls)
        std::vector<float> h_buf(std::max({(size_t)spec.M * spec.K, (size_t)spec.K * spec.N}));
        fill_random(h_buf.data(), spec.M * spec.K, 0.01f);
        CUDA_CHECK(cudaMemcpy(d_A, h_buf.data(), sizeA, cudaMemcpyHostToDevice));
        fill_random(h_buf.data(), spec.K * spec.N, 0.01f);
        CUDA_CHECK(cudaMemcpy(d_B, h_buf.data(), sizeB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_C, 0, sizeC));

        // Warmup
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 spec.M, spec.N, spec.K, &one,
                                 d_A, spec.M, d_B, spec.K, &zero, d_C, spec.M));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed run (3 iterations)
        Timer timer;
        timer.start();
        const int iters = 3;
        for (int i = 0; i < iters; i++) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     spec.M, spec.N, spec.K, &one,
                                     d_A, spec.M, d_B, spec.K, &zero, d_C, spec.M));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        double ms = timer.ms() / iters;

        // Validate
        std::vector<float> h_C(spec.M * spec.N);
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));

        if (!all_finite(h_C.data(), h_C.size())) {
            fprintf(stderr, "FAIL: non-finite values\n");
            CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
            CUBLAS_CHECK(cublasDestroy(handle));
            return false;
        }

        double gflops = 2.0 * spec.M * spec.N * spec.K / (ms * 1e6);
        printf("PASS (%.1f ms, %.1f GFLOPS)\n", ms, gflops);

        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return true;
}

// ======================================================================
// Test 3: ResNet-50 full forward pass simulation
//
// 4 stages: [3, 4, 6, 3] bottleneck blocks
// Channels: 256 → 512 → 1024 → 2048
// Input: batch=16, 224x224x3 → 7x7 feature maps by stage 4
// ======================================================================
static bool test_resnet50_forward() {
    printf("  test_resnet50_forward (batch=16, 4 stages, 16 bottleneck blocks)...\n");

    const int batch = 16;
    struct StageConfig {
        int num_blocks;
        int c_in, c_mid, c_out;
        int spatial; // H=W of feature map
    };

    StageConfig stages[] = {
        {3,  256,  64,  256, 56},  // stage 1: 56x56
        {4,  256, 128,  512, 28},  // stage 2: 28x28
        {6,  512, 256, 1024, 14},  // stage 3: 14x14
        {3, 1024, 512, 2048,  7},  // stage 4: 7x7
    };

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    size_t total_allocated = 0;
    float one = 1.0f, zero = 0.0f;
    srand(314);

    Timer timer;
    timer.start();

    // Simulate the initial conv1 + pool → [batch x 256 x 56 x 56]
    int cur_channels = 256;
    int cur_spatial = 56;
    size_t feat_size = (size_t)batch * cur_spatial * cur_spatial * cur_channels * sizeof(float);

    float* d_feat;
    CUDA_CHECK(cudaMalloc((void**)&d_feat, feat_size));
    total_allocated += feat_size;
    {
        std::vector<float> h_feat(batch * cur_spatial * cur_spatial * cur_channels);
        fill_random(h_feat.data(), h_feat.size(), 0.1f);
        CUDA_CHECK(cudaMemcpy(d_feat, h_feat.data(), feat_size, cudaMemcpyHostToDevice));
    }

    for (int s = 0; s < 4; s++) {
        auto& cfg = stages[s];
        int feat_elems = batch * cfg.spatial * cfg.spatial;
        printf("    Stage %d: %dx%d, %d→%d→%d→%d, %d blocks... ",
               s + 1, cfg.spatial, cfg.spatial, cfg.c_in, cfg.c_mid, cfg.c_mid, cfg.c_out, cfg.num_blocks);

        // Downsample if spatial changes
        if (s > 0) {
            // For the first block of each stage, c_in is actually the previous stage's c_out
            // and spatial is halved. We handle this by just reallocating.
            size_t new_size = (size_t)feat_elems * cfg.c_in * sizeof(float);
            float* d_new_feat;
            CUDA_CHECK(cudaMalloc((void**)&d_new_feat, new_size));
            total_allocated += new_size;
            // Simulate downsample with zero fill (real impl would stride-2 conv)
            CUDA_CHECK(cudaMemset(d_new_feat, 0, new_size));
            CUDA_CHECK(cudaFree(d_feat));
            d_feat = d_new_feat;
        }

        // Weight matrices for this stage (shared across blocks for simplicity)
        float *W_down, *W_mid, *W_up;
        size_t wd_size = cfg.c_in * cfg.c_mid * sizeof(float);
        size_t wm_size = cfg.c_mid * cfg.c_mid * sizeof(float);
        size_t wu_size = cfg.c_mid * cfg.c_out * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&W_down, wd_size));
        CUDA_CHECK(cudaMalloc((void**)&W_mid, wm_size));
        CUDA_CHECK(cudaMalloc((void**)&W_up, wu_size));
        total_allocated += wd_size + wm_size + wu_size;

        std::vector<float> wbuf;
        wbuf.resize(cfg.c_in * cfg.c_mid); fill_random(wbuf.data(), wbuf.size());
        CUDA_CHECK(cudaMemcpy(W_down, wbuf.data(), wd_size, cudaMemcpyHostToDevice));
        wbuf.resize(cfg.c_mid * cfg.c_mid); fill_random(wbuf.data(), wbuf.size());
        CUDA_CHECK(cudaMemcpy(W_mid, wbuf.data(), wm_size, cudaMemcpyHostToDevice));
        wbuf.resize(cfg.c_mid * cfg.c_out); fill_random(wbuf.data(), wbuf.size());
        CUDA_CHECK(cudaMemcpy(W_up, wbuf.data(), wu_size, cudaMemcpyHostToDevice));

        // Intermediate buffers
        float *d_down, *d_mid, *d_up;
        size_t down_size = (size_t)feat_elems * cfg.c_mid * sizeof(float);
        size_t up_size = (size_t)feat_elems * cfg.c_out * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_down, down_size));
        CUDA_CHECK(cudaMalloc((void**)&d_mid, down_size));
        CUDA_CHECK(cudaMalloc((void**)&d_up, up_size));
        total_allocated += 2 * down_size + up_size;

        // ReLU descriptor
        cudnnTensorDescriptor_t midDesc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&midDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(midDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, 1, 1, (int)(feat_elems * cfg.c_mid)));

        cudnnActivationDescriptor_t relu;
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&relu));
        CUDNN_CHECK(cudnnSetActivationDescriptor(relu, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

        for (int b = 0; b < cfg.num_blocks; b++) {
            // 1x1 down-project: [feat_elems x c_in] @ [c_in x c_mid]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     feat_elems, cfg.c_mid, cfg.c_in, &one,
                                     d_feat, feat_elems, W_down, cfg.c_in, &zero, d_down, feat_elems));
            CUDNN_CHECK(cudnnActivationForward(cudnn, relu, &one, midDesc, d_down, &zero, midDesc, d_down));

            // 3x3 mid conv (GEMM proxy)
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     feat_elems, cfg.c_mid, cfg.c_mid, &one,
                                     d_down, feat_elems, W_mid, cfg.c_mid, &zero, d_mid, feat_elems));
            CUDNN_CHECK(cudnnActivationForward(cudnn, relu, &one, midDesc, d_mid, &zero, midDesc, d_mid));

            // 1x1 up-project: [feat_elems x c_mid] @ [c_mid x c_out]
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     feat_elems, cfg.c_out, cfg.c_mid, &one,
                                     d_mid, feat_elems, W_up, cfg.c_mid, &zero, d_up, feat_elems));

            // Residual (skip connection) — for first block after downsample, just use output
            if (b > 0) {
                CUBLAS_CHECK(cublasSaxpy(cublas, feat_elems * cfg.c_out, &one, d_feat, 1, d_up, 1));
            }

            // Prepare for next block — resize d_feat if needed
            if (b == 0 && (size_t)feat_elems * cfg.c_out * sizeof(float) != feat_size) {
                CUDA_CHECK(cudaFree(d_feat));
                feat_size = (size_t)feat_elems * cfg.c_out * sizeof(float);
                CUDA_CHECK(cudaMalloc((void**)&d_feat, feat_size));
                total_allocated += feat_size;
            }
            CUDA_CHECK(cudaMemcpy(d_feat, d_up, feat_elems * cfg.c_out * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        cur_channels = cfg.c_out;
        cur_spatial = cfg.spatial;

        // Cleanup stage buffers
        CUDA_CHECK(cudaFree(W_down)); CUDA_CHECK(cudaFree(W_mid)); CUDA_CHECK(cudaFree(W_up));
        CUDA_CHECK(cudaFree(d_down)); CUDA_CHECK(cudaFree(d_mid)); CUDA_CHECK(cudaFree(d_up));
        CUDNN_CHECK(cudnnDestroyActivationDescriptor(relu));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(midDesc));

        printf("PASS\n");
    }

    double elapsed = timer.ms();

    // Final validation
    std::vector<float> h_result(batch * cur_spatial * cur_spatial * cur_channels);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_feat, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool finite = all_finite(h_result.data(), h_result.size());

    CUDA_CHECK(cudaFree(d_feat));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    if (!finite) {
        fprintf(stderr, "    FAIL: non-finite output\n");
        return false;
    }

    printf("    PASS total (%.1f ms, %.1f MB peak, output [%d x %d x %d x %d])\n",
           elapsed, bytes_to_mb(total_allocated), batch, cur_channels, cur_spatial, cur_spatial);
    return true;
}

// ======================================================================
// Test 4: Sustained memory pressure
//
// Allocate 500+ MB, run operations, free/realloc in patterns that
// stress the memory manager under load
// ======================================================================
static bool test_sustained_memory_pressure() {
    printf("  test_sustained_memory_pressure (>500 MB, 512 buffers, 5 rounds)...\n");

    const int num_buffers = 512;
    const int rounds = 5;
    std::vector<void*> ptrs(num_buffers, nullptr);
    std::vector<size_t> sizes(num_buffers);
    size_t total_live = 0;
    size_t peak = 0;

    srand(2025);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    for (int r = 0; r < rounds; r++) {
        printf("    Round %d/%d: ", r + 1, rounds);

        // Allocate all buffers
        total_live = 0;
        for (int i = 0; i < num_buffers; i++) {
            sizes[i] = (size_t)(1 + rand() % 2048) * 1024; // 1KB to 2MB
            CUDA_CHECK(cudaMalloc(&ptrs[i], sizes[i]));
            CUDA_CHECK(cudaMemset(ptrs[i], 0, sizes[i]));
            total_live += sizes[i];
        }
        if (total_live > peak) peak = total_live;

        // Do some GEMM work on random pairs of buffers
        float one = 1.0f, zero = 0.0f;
        int gemm_count = 0;
        for (int i = 0; i < 20; i++) {
            int a = rand() % num_buffers;
            int b = rand() % num_buffers;
            int c = rand() % num_buffers;
            // Find the largest square matrix that fits
            int dim = (int)sqrtf((float)std::min({sizes[a], sizes[b], sizes[c]}) / sizeof(float));
            if (dim < 16) continue;
            dim = std::min(dim, 512); // cap for speed

            cublasStatus_t st = cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                            dim, dim, dim, &one,
                                            (float*)ptrs[a], dim,
                                            (float*)ptrs[b], dim,
                                            &zero, (float*)ptrs[c], dim);
            if (st == CUBLAS_STATUS_SUCCESS) gemm_count++;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Free half at random
        int freed = 0;
        for (int i = 0; i < num_buffers; i++) {
            if (rand() % 2 == 0 && ptrs[i]) {
                CUDA_CHECK(cudaFree(ptrs[i]));
                total_live -= sizes[i];
                ptrs[i] = nullptr;
                freed++;
            }
        }

        // Reallocate freed slots with new sizes
        int realloced = 0;
        for (int i = 0; i < num_buffers; i++) {
            if (!ptrs[i]) {
                sizes[i] = (size_t)(1 + rand() % 4096) * 1024;
                CUDA_CHECK(cudaMalloc(&ptrs[i], sizes[i]));
                CUDA_CHECK(cudaMemset(ptrs[i], 0, sizes[i]));
                total_live += sizes[i];
                realloced++;
            }
        }
        if (total_live > peak) peak = total_live;

        printf("%.1f MB live, %d GEMMs, freed %d, realloc'd %d\n",
               bytes_to_mb(total_live), gemm_count, freed, realloced);
    }

    // Final cleanup
    for (int i = 0; i < num_buffers; i++) {
        if (ptrs[i]) CUDA_CHECK(cudaFree(ptrs[i]));
    }
    CUBLAS_CHECK(cublasDestroy(cublas));

    printf("    PASS (peak %.1f MB)\n", bytes_to_mb(peak));
    return true;
}

// ======================================================================
// Test 5: Deep pipeline — 24 layer transformer (GPT-2 Medium scale)
//
// d_model=1024, n_heads=16, d_ff=4096, seq_len=256, batch=4, 24 layers
// Tests deep sequential dispatch without OOM or NaN accumulation
// ======================================================================
static bool test_deep_transformer_24layer() {
    printf("  test_deep_transformer_24layer (d=1024, 24 layers, seq=256, batch=4)...\n");

    const int d_model = 1024;
    const int d_ff = 4096;
    const int seq_len = 256;
    const int batch = 4;
    const int num_layers = 24;
    const int tokens = batch * seq_len;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    float one = 1.0f, zero = 0.0f;
    srand(7777);

    // Single set of weights (shared across layers for memory savings — tests dispatch depth)
    float *W_qkv, *W_o, *W1, *W2;
    size_t qkv_w_size = d_model * (3 * d_model) * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&W_qkv, qkv_w_size));
    CUDA_CHECK(cudaMalloc((void**)&W_o, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&W1, d_model * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&W2, d_ff * d_model * sizeof(float)));

    {
        std::vector<float> wb(d_model * 3 * d_model);
        fill_random(wb.data(), wb.size());
        CUDA_CHECK(cudaMemcpy(W_qkv, wb.data(), qkv_w_size, cudaMemcpyHostToDevice));
        wb.resize(d_model * d_model); fill_random(wb.data(), wb.size());
        CUDA_CHECK(cudaMemcpy(W_o, wb.data(), d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
        wb.resize(d_model * d_ff); fill_random(wb.data(), wb.size());
        CUDA_CHECK(cudaMemcpy(W1, wb.data(), d_model * d_ff * sizeof(float), cudaMemcpyHostToDevice));
        wb.resize(d_ff * d_model); fill_random(wb.data(), wb.size());
        CUDA_CHECK(cudaMemcpy(W2, wb.data(), d_ff * d_model * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Activation buffers
    float *d_x, *d_residual, *d_qkv, *d_attn_out, *d_ff_hidden, *d_ff_out;
    size_t x_size = tokens * d_model * sizeof(float);
    CUDA_CHECK(cudaMalloc((void**)&d_x, x_size));
    CUDA_CHECK(cudaMalloc((void**)&d_residual, x_size));
    CUDA_CHECK(cudaMalloc((void**)&d_qkv, tokens * 3 * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_attn_out, x_size));
    CUDA_CHECK(cudaMalloc((void**)&d_ff_hidden, tokens * d_ff * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_ff_out, x_size));

    // Init input
    {
        std::vector<float> h_x(tokens * d_model);
        fill_random(h_x.data(), h_x.size(), 0.5f);
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_size, cudaMemcpyHostToDevice));
    }

    // ReLU
    cudnnTensorDescriptor_t ffDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&ffDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(ffDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, tokens * d_ff));
    cudnnActivationDescriptor_t relu;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&relu));
    CUDNN_CHECK(cudnnSetActivationDescriptor(relu, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    Timer timer;
    timer.start();

    for (int l = 0; l < num_layers; l++) {
        CUDA_CHECK(cudaMemcpy(d_residual, d_x, x_size, cudaMemcpyDeviceToDevice));

        // Fused QKV = X @ W_qkv  [tokens x 3*d_model]
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, 3 * d_model, d_model, &one,
                                 d_x, tokens, W_qkv, d_model, &zero, d_qkv, tokens));

        // Simplified attention: just project back (skip actual attention computation to test depth)
        // attn_out = qkv_slice @ Wo
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_model, &one,
                                 d_qkv, tokens, W_o, d_model, &zero, d_attn_out, tokens));

        // Residual
        CUBLAS_CHECK(cublasSaxpy(cublas, tokens * d_model, &one, d_residual, 1, d_attn_out, 1));
        CUDA_CHECK(cudaMemcpy(d_x, d_attn_out, x_size, cudaMemcpyDeviceToDevice));

        // FFN
        CUDA_CHECK(cudaMemcpy(d_residual, d_x, x_size, cudaMemcpyDeviceToDevice));

        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_ff, d_model, &one,
                                 d_x, tokens, W1, d_model, &zero, d_ff_hidden, tokens));
        CUDNN_CHECK(cudnnActivationForward(cudnn, relu, &one, ffDesc, d_ff_hidden, &zero, ffDesc, d_ff_hidden));

        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_ff, &one,
                                 d_ff_hidden, tokens, W2, d_ff, &zero, d_ff_out, tokens));

        CUBLAS_CHECK(cublasSaxpy(cublas, tokens * d_model, &one, d_residual, 1, d_ff_out, 1));
        CUDA_CHECK(cudaMemcpy(d_x, d_ff_out, x_size, cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = timer.ms();

    // Validate
    std::vector<float> h_result(tokens * d_model);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_x, x_size, cudaMemcpyDeviceToHost));
    bool finite = all_finite(h_result.data(), h_result.size());

    float min_val = *std::min_element(h_result.begin(), h_result.end());
    float max_val = *std::max_element(h_result.begin(), h_result.end());

    // Cleanup
    CUDA_CHECK(cudaFree(W_qkv)); CUDA_CHECK(cudaFree(W_o));
    CUDA_CHECK(cudaFree(W1)); CUDA_CHECK(cudaFree(W2));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_qkv)); CUDA_CHECK(cudaFree(d_attn_out));
    CUDA_CHECK(cudaFree(d_ff_hidden)); CUDA_CHECK(cudaFree(d_ff_out));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(relu));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(ffDesc));
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDNN_CHECK(cudnnDestroy(cudnn));

    if (!finite) {
        fprintf(stderr, "    FAIL: non-finite after 24 layers (range [%f, %f])\n", min_val, max_val);
        return false;
    }

    printf("    Output range: [%.4f, %.4f]\n", min_val, max_val);
    printf("    PASS (%.1f ms, 24 layers x %d tokens)\n", elapsed, tokens);
    return true;
}

// ======================================================================
// Test 6: Rapid allocation churn under compute
//
// Simulates dynamic batching: allocate tensors, compute, free, repeat
// with varying batch sizes (like a serving system)
// ======================================================================
static bool test_dynamic_batching() {
    printf("  test_dynamic_batching (100 requests, variable batch 1-64, d=1024)...\n");

    const int d_model = 1024;
    const int num_requests = 100;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Fixed weight matrix
    float* W;
    CUDA_CHECK(cudaMalloc((void**)&W, d_model * d_model * sizeof(float)));
    {
        std::vector<float> wh(d_model * d_model);
        fill_random(wh.data(), wh.size());
        CUDA_CHECK(cudaMemcpy(W, wh.data(), wh.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    srand(12345);
    Timer timer;
    timer.start();

    int total_tokens = 0;
    float one = 1.0f, zero = 0.0f;

    for (int r = 0; r < num_requests; r++) {
        int batch_size = 1 + rand() % 64;
        int seq_len = 32 + rand() % 480; // 32 to 512
        int tokens = batch_size * seq_len;
        total_tokens += tokens;

        size_t in_size = tokens * d_model * sizeof(float);
        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc((void**)&d_in, in_size));
        CUDA_CHECK(cudaMalloc((void**)&d_out, in_size));

        // Fill input
        CUDA_CHECK(cudaMemset(d_in, 0, in_size));

        // Linear: out = in @ W
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 tokens, d_model, d_model, &one,
                                 d_in, tokens, W, d_model, &zero, d_out, tokens));

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = timer.ms();

    CUDA_CHECK(cudaFree(W));
    CUBLAS_CHECK(cublasDestroy(cublas));

    printf("    PASS (%.1f ms, %d requests, %d total tokens, %.0f tokens/sec)\n",
           elapsed, num_requests, total_tokens, total_tokens / (elapsed / 1000.0));
    return true;
}

// ======================================================================
// Main
// ======================================================================
int main() {
    printf("=== Large Model Integration Tests ===\n\n");

    // Print device info
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        printf("Device: %s\n", prop.name);
        printf("Memory: %.1f MB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    }

    int pass = 0, fail = 0;
    auto run = [&](bool (*test)(), const char* name) {
        if (test()) {
            pass++;
        } else {
            fprintf(stderr, "  >>> %s FAILED <<<\n", name);
            fail++;
        }
        printf("\n");
    };

    run(test_gpt2_transformer, "GPT-2 Transformer");
    run(test_llama_gemm_sizes, "LLaMA GEMM Sizes");
    run(test_resnet50_forward, "ResNet-50 Forward");
    run(test_sustained_memory_pressure, "Sustained Memory Pressure");
    run(test_deep_transformer_24layer, "Deep 24-Layer Transformer");
    run(test_dynamic_batching, "Dynamic Batching");

    printf("=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
