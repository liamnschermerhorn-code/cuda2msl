#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cuda_metal {

// Pattern-based kernel fusion engine.
//
// Intercepts kernel dispatch calls, maintains a sliding window of
// recent kernel names, and fuses consecutive kernels that match
// registered patterns into single fused dispatches.
//
// Built-in fusions:
//   layernorm + gelu        -> fused_layernorm_gelu
//   layernorm + add         -> fused_layernorm_add
//   gelu * mul              -> fused_gelu_mul
//   silu * mul              -> fused_silu_mul
//   rope + attention        -> fused_rope_attention
//   silu + mul + quantize   -> fused_silu_mul_quantize

struct FusionPattern {
    std::vector<std::string> kernel_sequence;  // e.g. {"layernorm", "gelu"}
    std::string              fused_name;        // e.g. "fused_layernorm_gelu"
};

// Callback for executing a fused kernel.
using FusedKernelFn = std::function<void(
    MTL::ComputeCommandEncoder* encoder,
    const std::vector<MTL::Buffer*>& buffers,
    size_t num_elements
)>;

class KernelFusion {
public:
    static KernelFusion& instance();

    // Register a fusion pattern and its execution function.
    void registerPattern(const FusionPattern& pattern, FusedKernelFn fn);

    // Check if the recent kernel history matches any fusion pattern.
    // Returns the fused kernel name, or empty string if no match.
    std::string checkFusion(const std::string& kernel_name);

    // Get the fused kernel function for a given fused name.
    FusedKernelFn getFusedKernel(const std::string& fused_name) const;

    // Clear history (e.g. between inference batches).
    void clearHistory();

    // Enable/disable fusion (useful for debugging).
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    // Statistics
    size_t fusionCount() const { return fusion_count_; }

    KernelFusion(const KernelFusion&) = delete;
    KernelFusion& operator=(const KernelFusion&) = delete;

private:
    KernelFusion();

    static constexpr size_t kMaxHistory = 8;

    std::vector<FusionPattern> patterns_;
    std::unordered_map<std::string, FusedKernelFn> fused_kernels_;
    std::vector<std::string> history_;

    bool   enabled_ = true;
    size_t fusion_count_ = 0;
    mutable std::mutex mutex_;

    void registerBuiltins();
};

} // namespace cuda_metal
