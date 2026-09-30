#pragma once

// CUDA-to-Metal Translation Layer
// Flash Attention forward and backward passes
//
// Initial implementation uses CPU-side computation on unified memory
// (Metal shared storage), matching the pattern of existing BLAS-1 ops.
// Metal compute shader optimization can be added later.

#include "cuda.h"

#ifdef __cplusplus

struct FlashAttnParams {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float softmax_scale;
    bool is_causal;
};

// Flash Attention forward pass.
//
// Inputs:
//   Q, K, V  - [batch_size, num_heads, seq_len, head_dim] row-major
// Outputs:
//   O         - [batch_size, num_heads, seq_len, head_dim] row-major
//   softmax_lse - [batch_size, num_heads, seq_len] log-sum-exp per row (for backward)
//
// Uses tiled online softmax (FlashAttention-2 algorithm).
cudaError_t flash_attn_forward(
    const float* Q, const float* K, const float* V, float* O,
    float* softmax_lse,
    const FlashAttnParams& params, cudaStream_t stream);

// Flash Attention backward pass.
//
// Inputs:
//   dO         - gradient of output [batch_size, num_heads, seq_len, head_dim]
//   Q, K, V    - original inputs from forward
//   O          - forward output
//   softmax_lse - log-sum-exp from forward
// Outputs:
//   dQ, dK, dV - gradients [batch_size, num_heads, seq_len, head_dim]
//
// Recomputes attention during backward (memory-efficient).
cudaError_t flash_attn_backward(
    const float* dO, const float* Q, const float* K, const float* V,
    const float* O, const float* softmax_lse,
    float* dQ, float* dK, float* dV,
    const FlashAttnParams& params, cudaStream_t stream);

#endif // __cplusplus
