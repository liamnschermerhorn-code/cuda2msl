#pragma once

#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

#include "cuda.h"

namespace cuda_metal {

/// Tensor metadata passed from PyTorch to the Metal dispatch layer.
/// This is the bridge type — PyTorch fills it in, Metal dispatch reads it.
struct TensorMeta {
    void*    data_ptr = nullptr;     // Device pointer (MTL::Buffer contents)
    int64_t  numel    = 0;           // Total number of elements
    int      ndim     = 0;           // Number of dimensions
    int64_t  sizes[8]   = {};        // Shape (max 8 dims)
    int64_t  strides[8] = {};        // Strides
    int      dtype    = 0;           // 0=float32, 1=float16, 2=bfloat16, 3=int32, 4=int64, 5=int8, 6=uint8, 7=bool
    size_t   element_size = 4;       // Bytes per element
};

/// Signature for an aten:: operator implementation backed by Metal
using AtenOpFn = std::function<cudaError_t(
    const std::vector<TensorMeta>& inputs,
    const std::vector<TensorMeta>& outputs,
    const void* extra_args,
    cudaStream_t stream)>;

/// PyTorchBridge — maps aten:: operator names to Metal kernel dispatch.
///
/// PyTorch's CUDA backend calls into this bridge when running on our
/// translation layer. The bridge looks up the operator, maps tensor
/// metadata to Metal buffer arguments, and dispatches via KernelDispatch.
class PyTorchBridge {
public:
    static PyTorchBridge& instance();

    /// Register an aten:: operator implementation
    void registerOp(const std::string& op_name, AtenOpFn fn);

    /// Check if an operator is registered
    bool hasOp(const std::string& op_name) const;

    /// Dispatch an aten:: operator
    cudaError_t dispatch(
        const std::string& op_name,
        const std::vector<TensorMeta>& inputs,
        const std::vector<TensorMeta>& outputs,
        const void* extra_args = nullptr,
        cudaStream_t stream = nullptr);

    /// List all registered ops
    std::vector<std::string> listOps() const;

    /// Initialize all built-in operator mappings
    void registerBuiltinOps();

    PyTorchBridge(const PyTorchBridge&) = delete;
    PyTorchBridge& operator=(const PyTorchBridge&) = delete;

private:
    PyTorchBridge() = default;

    std::unordered_map<std::string, AtenOpFn> ops_;
    mutable std::mutex mutex_;
};

} // namespace cuda_metal
