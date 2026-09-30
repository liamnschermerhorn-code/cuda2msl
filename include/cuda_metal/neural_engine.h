#pragma once

// Neural Engine / AMX Accelerator Bridge
//
// All Apple Silicon: offloads matrix multiply, normalization, and activation
// functions to the CPU's AMX coprocessor via Accelerate/BNNS. The AMX provides
// dedicated matrix multiply hardware on the CPU die that operates in parallel
// with the GPU. Available on all Apple Silicon (M1+), not just M5+.

#include <cstddef>
#include <cstdint>

namespace cuda_metal {

// Initialize BNNS/AMX engine. Must be called after device tier detection.
void neural_engine_init();
void neural_engine_shutdown();

// AMX-accelerated matrix multiply (M5+ only, falls back to false on legacy)
// C[M,N] = A[M,K] @ B[K,N]  (row-major)
// Returns true if executed on AMX, false if caller should use MPS/CPU path.
bool neural_engine_matmul_f32(const float* A, const float* B, float* C,
                               int M, int N, int K);

// AMX-accelerated fp16 matmul with fp32 accumulation
// A is fp16 [M,K], B is fp16 [K,N], C is fp32 [M,N]
bool neural_engine_matmul_f16_f32(const uint16_t* A, const uint16_t* B, float* C,
                                   int M, int N, int K);

// AMX-accelerated RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
bool neural_engine_rmsnorm(float* out, const float* x, const float* w,
                            int n, float eps);

// AMX-accelerated softmax in-place
bool neural_engine_softmax(float* x, int n);

// AMX-accelerated SiLU(gate) * up element-wise
bool neural_engine_silu_mul(float* gate, const float* up, int n);

// AMX-accelerated RoPE (rotary position embedding)
bool neural_engine_rope(float* q, float* k, int head_dim, int n_heads,
                         int n_kv_heads, int pos, float theta);

} // namespace cuda_metal
