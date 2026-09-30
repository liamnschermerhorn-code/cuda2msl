#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>

namespace cuda_metal {

// Quantization formats
enum class QuantFormat : int {
    INT4   = 0,  // 4-bit integer (GPTQ/AWQ style)
    INT8   = 1,  // 8-bit integer
    FP8_E4M3 = 2,  // 8-bit float, 4-bit exponent, 3-bit mantissa
    FP8_E5M2 = 3,  // 8-bit float, 5-bit exponent, 2-bit mantissa
};

struct QuantParams {
    float  scale;      // Dequantization scale
    float  zero_point; // Dequantization zero point
    int    group_size;  // Quantization group size (e.g. 128)
};

// Quantization pipeline for inference weight compression.
//
// Supports:
//  - Quantize/dequantize in INT4, INT8, FP8_E4M3, FP8_E5M2
//  - Fused dequant + matmul (no intermediate float buffer)
//  - W4A16 GEMM: 4-bit weights, 16-bit activations (GPTQ/AWQ pattern)

class Quantization {
public:
    static Quantization& instance();

    // Initialize Metal pipelines for quantization ops.
    bool init();

    // Quantize float data into the given format.
    // dst must be pre-allocated with appropriate size.
    void quantize(
        MTL::ComputeCommandEncoder* encoder,
        MTL::Buffer* src,       // float input
        MTL::Buffer* dst,       // quantized output
        MTL::Buffer* scales,    // per-group scales
        size_t num_elements,
        QuantFormat format,
        int group_size = 128
    );

    // Dequantize into float.
    void dequantize(
        MTL::ComputeCommandEncoder* encoder,
        MTL::Buffer* src,       // quantized input
        MTL::Buffer* dst,       // float output
        MTL::Buffer* scales,    // per-group scales
        size_t num_elements,
        QuantFormat format,
        int group_size = 128
    );

    // Fused dequantize + GEMM: C = A * dequant(B)
    // A is half/float, B is quantized weights.
    void quantizedGemm(
        MTL::ComputeCommandEncoder* encoder,
        MTL::Buffer* A,         // Activations (half or float)
        MTL::Buffer* B,         // Quantized weights
        MTL::Buffer* scales,    // Weight scales
        MTL::Buffer* C,         // Output
        size_t M, size_t N, size_t K,
        QuantFormat format,
        int group_size = 128
    );

    // W4A16 GEMM: 4-bit weights, 16-bit activations (most common LLM pattern)
    void w4a16Gemm(
        MTL::ComputeCommandEncoder* encoder,
        MTL::Buffer* activations,  // half [M, K]
        MTL::Buffer* weights,      // int4 packed [K, N/2]
        MTL::Buffer* scales,       // half [K/group_size, N]
        MTL::Buffer* zeros,        // half [K/group_size, N]
        MTL::Buffer* output,       // half [M, N]
        size_t M, size_t N, size_t K,
        int group_size = 128
    );

    bool isInitialized() const { return initialized_; }

    Quantization(const Quantization&) = delete;
    Quantization& operator=(const Quantization&) = delete;

private:
    Quantization() = default;

    MTL::ComputePipelineState* quant_int4_pipeline_   = nullptr;
    MTL::ComputePipelineState* quant_int8_pipeline_   = nullptr;
    MTL::ComputePipelineState* quant_fp8_pipeline_    = nullptr;
    MTL::ComputePipelineState* dequant_int4_pipeline_ = nullptr;
    MTL::ComputePipelineState* dequant_int8_pipeline_ = nullptr;
    MTL::ComputePipelineState* dequant_fp8_pipeline_  = nullptr;
    MTL::ComputePipelineState* gemm_int4_pipeline_    = nullptr;
    MTL::ComputePipelineState* gemm_int8_pipeline_    = nullptr;
    MTL::ComputePipelineState* w4a16_pipeline_        = nullptr;

    bool initialized_ = false;
};

} // namespace cuda_metal
