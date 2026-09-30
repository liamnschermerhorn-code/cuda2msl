#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace cuda_metal {

// Threadgroup size tuner for Apple Silicon's 32-wide SIMD groups.
//
// Apple GPUs execute in 32-thread SIMD groups. Optimal threadgroup sizes
// must be multiples of 32, and the maximum depends on the pipeline's
// register/shared memory usage. This class queries the pipeline state
// and returns tuned sizes.

struct ThreadgroupConfig {
    size_t threads_per_group;   // Total threads per threadgroup
    size_t groups_per_grid;     // Total threadgroups to dispatch
    size_t threadgroup_memory;  // Threadgroup memory bytes (0 if none)
};

class ThreadgroupTuner {
public:
    static ThreadgroupTuner& instance();

    // Query the pipeline for optimal threadgroup size, given total work items
    // and shared memory requirement.
    ThreadgroupConfig optimal(
        MTL::ComputePipelineState* pipeline,
        size_t total_threads,
        size_t shared_mem_bytes = 0
    );

    // Presets for common kernel patterns
    ThreadgroupConfig forElementwise(size_t numel);
    ThreadgroupConfig forReduction(size_t numel);
    ThreadgroupConfig forMatmul(size_t M, size_t N, size_t K);
    ThreadgroupConfig forAttention(size_t seq_len, size_t head_dim);

    // Clear the cache (e.g. after pipeline recreation)
    void clearCache();

    ThreadgroupTuner(const ThreadgroupTuner&) = delete;
    ThreadgroupTuner& operator=(const ThreadgroupTuner&) = delete;

private:
    ThreadgroupTuner() = default;

    static constexpr size_t kSimdWidth = 32;

    // Round up to multiple of SIMD width
    static size_t roundToSimd(size_t n) {
        return ((n + kSimdWidth - 1) / kSimdWidth) * kSimdWidth;
    }

    // Ceiling division
    static size_t ceilDiv(size_t a, size_t b) {
        return (a + b - 1) / b;
    }

    // Cache: pipeline pointer -> max threads per threadgroup
    std::unordered_map<const void*, size_t> cache_;
    mutable std::mutex mutex_;
};

} // namespace cuda_metal
