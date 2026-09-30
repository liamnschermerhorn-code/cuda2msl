#pragma once

#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <vector>

#include "cuda.h"

namespace cuda_metal {

/// Type tag for kernel arguments
enum class KernelArgType {
    DevicePointer,  // Pointer to a Metal buffer allocation
    Scalar,         // Raw scalar value (int, float, etc.)
    Struct,         // Raw struct bytes
    Buffer          // Pre-resolved MTL::Buffer with offset
};

/// A single kernel argument with type information
struct KernelArg {
    KernelArgType type = KernelArgType::Scalar;
    const void*   data = nullptr;       // Pointer to the argument data
    size_t        size = 0;             // Size of the argument in bytes
    void*         metal_buffer = nullptr; // For KernelArgType::Buffer
    size_t        buffer_offset = 0;      // For KernelArgType::Buffer
};

/// KernelDispatch — high-level kernel dispatch that handles argument marshaling,
/// threadgroup memory allocation, and grid/threadgroup size computation.
///
/// Sits between PyTorch's operator dispatch and the raw Metal compute encoder.
class KernelDispatch {
public:
    static KernelDispatch& instance();

    /// Dispatch a kernel with typed arguments
    cudaError_t dispatch(
        const std::string& kernel_name,
        dim3 grid_dim,
        dim3 block_dim,
        const std::vector<KernelArg>& args,
        size_t shared_mem_bytes = 0,
        cudaStream_t stream = nullptr);

    /// Dispatch with a pre-created pipeline (bypasses ShaderLoader lookup).
    /// Used for specialized pipelines with function constants.
    cudaError_t dispatchWithPipeline(
        MTL::ComputePipelineState* pipeline,
        dim3 grid_dim,
        dim3 block_dim,
        const std::vector<KernelArg>& args,
        size_t shared_mem_bytes = 0,
        cudaStream_t stream = nullptr);

    /// Dispatch a kernel by name with raw argument pointers + sizes.
    /// Automatically classifies arguments as device pointers vs scalars.
    cudaError_t dispatchByName(
        const std::string& kernel_name,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z,
        void** raw_args, size_t num_args,
        const size_t* arg_sizes,
        size_t shared_mem_bytes = 0,
        cudaStream_t stream = nullptr);

    KernelDispatch(const KernelDispatch&) = delete;
    KernelDispatch& operator=(const KernelDispatch&) = delete;

private:
    KernelDispatch() = default;
    ~KernelDispatch() = default;
};

} // namespace cuda_metal
