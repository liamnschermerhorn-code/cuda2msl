#pragma once

// Device Tier Classification
//
// Two processing tiers based on Apple Silicon generation:
//   M5+   : Tensor cores (16x16 simdgroup), Neural Engine offload, function-constant
//           specialized shaders, smart KV cache paging, binary-obfuscated shader libraries
//   Legacy: Standard MPS GEMM (8x8 simdgroup), CPU elementwise ops, basic KV cache

#include <cstddef>
#include <cstdint>
#include <string>

namespace cuda_metal {

enum class DeviceTier {
    LEGACY,   // M1, M2, M3, M4
    M5_PLUS   // M5 and later
};

struct DeviceCapabilities {
    DeviceTier tier = DeviceTier::LEGACY;
    std::string chip_name;
    int chip_generation = 0;  // 1=M1, 2=M2, ..., 5=M5

    // Feature flags
    bool has_tensor_cores_16x16 = false; // Extended simdgroup_matrix (16x16 tiles)
    bool has_neural_engine      = false; // ANE for offload
    bool has_bf16_matmul        = false; // Native BF16 matrix multiply
    bool has_int8_matmul        = false; // Native INT8 matrix multiply
    bool has_sparse_matmul      = false; // Hardware sparse matrix ops
    bool has_function_pointers  = false; // Indirect compute dispatch
    bool has_mesh_shaders       = false; // Mesh/object shaders
    bool has_ray_tracing        = false; // RT accelerator

    // Memory
    size_t unified_memory_bytes = 0;
    size_t max_buffer_bytes     = 0;
    size_t recommended_working_set = 0;

    // Compute
    int gpu_core_count         = 0;
    int neural_engine_cores    = 0;
    int max_threads_per_tg     = 0;
    int simd_width             = 32;
    int max_simdgroup_matrix_dim = 8; // 8 for legacy, 16 for M5+
};

// Detect capabilities of the current device (call once at init).
// Implemented in device_tier.mm (needs ObjC for Metal queries).
DeviceCapabilities detect_device_capabilities();

// Get cached capabilities (call detect first).
const DeviceCapabilities& device_caps();

// Convenience
inline bool is_m5_plus() { return device_caps().tier == DeviceTier::M5_PLUS; }

} // namespace cuda_metal
