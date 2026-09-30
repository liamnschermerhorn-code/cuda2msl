#pragma once

#include "cuda.h"

namespace cuda_metal { namespace autograd {

// ---------------------------------------------------------------------------
// Backward kernels for common PyTorch ops.
//
// These operate on host-accessible unified memory (MTL::Buffer contents()),
// so they execute directly on the CPU.  For large tensors the Metal forward
// kernels do the heavy lifting; the backward pass through these helpers is
// fine for correctness and moderate workloads.
//
// All functions return cudaSuccess on success.
// ---------------------------------------------------------------------------

/// ReLU backward: grad_in[i] = input[i] > 0 ? grad_out[i] : 0
cudaError_t relu_backward(const float* grad_out, const float* input,
                          float* grad_in, size_t n);

/// Sigmoid backward: grad_in[i] = grad_out[i] * output[i] * (1 - output[i])
cudaError_t sigmoid_backward(const float* grad_out, const float* output,
                             float* grad_in, size_t n);

/// Tanh backward: grad_in[i] = grad_out[i] * (1 - output[i]^2)
cudaError_t tanh_backward(const float* grad_out, const float* output,
                          float* grad_in, size_t n);

/// GELU backward (tanh approximation):
///   phi = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///   grad_in = grad_out * (phi + x * dphi/dx)
cudaError_t gelu_backward(const float* grad_out, const float* input,
                          float* grad_in, size_t n);

/// SiLU (Swish) backward:
///   silu(x) = x * sigmoid(x)
///   grad_in = grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
cudaError_t silu_backward(const float* grad_out, const float* input,
                          float* grad_in, size_t n);

/// Softmax backward (last-dim):
///   grad_in[i,j] = output[i,j] * (grad_out[i,j] - dot(grad_out[i,:], output[i,:]))
cudaError_t softmax_backward(const float* grad_out, const float* output,
                             float* grad_in, int rows, int cols);

/// Layer-norm backward:
///   Three-pass algorithm computing grad_in, grad_weight, grad_bias.
///   mean/rstd are per-row statistics from the forward pass (length N).
///   Input/output shapes: [N, D].
cudaError_t layer_norm_backward(const float* grad_out, const float* input,
                                const float* weight,
                                const float* mean, const float* rstd,
                                float* grad_in, float* grad_weight,
                                float* grad_bias,
                                int N, int D);

/// Cross-entropy backward:
///   grad_in[i,c] = grad_out[i] * (probs[i,c] - one_hot(targets[i], c))
///   probs is [N, C], targets is [N] with values in [0, C).
cudaError_t cross_entropy_backward(const float* grad_out, const float* probs,
                                   const int* targets,
                                   float* grad_in, int N, int C);

// Note: matmul backward dispatches to cublasSgemm:
//   dA = dC @ B^T
//   dB = A^T @ dC
// This is handled in the registration layer, not here.

}} // namespace cuda_metal::autograd
