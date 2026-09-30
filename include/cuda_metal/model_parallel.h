#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <mutex>
#include <vector>

namespace cuda_metal {

// Single-GPU tensor parallelism via concurrent command queues.
//
// Splits large GEMMs across multiple command queues on the same GPU,
// using SharedEvents for synchronization. Apple Silicon GPUs have
// enough ALU clusters to benefit from queue-level parallelism for
// independent matrix partitions.

enum class ParallelMode {
    ROW_PARALLEL,  // Split rows of A across partitions
    COL_PARALLEL,  // Split columns of B across partitions
};

class ModelParallel {
public:
    static ModelParallel& instance();

    // Configure the number of partitions (concurrent queues).
    // Defaults to GPU core count / 256 (clamped to 2-8).
    bool configure(size_t num_partitions = 0);

    // Parallel GEMM: C = A * B, split across partitions.
    void parallelGemm(
        MTL::Buffer* A, MTL::Buffer* B, MTL::Buffer* C,
        size_t M, size_t N, size_t K,
        ParallelMode mode = ParallelMode::ROW_PARALLEL
    );

    // Synchronize all partitions.
    void sync();

    size_t numPartitions() const { return num_partitions_; }

    ModelParallel(const ModelParallel&) = delete;
    ModelParallel& operator=(const ModelParallel&) = delete;

private:
    ModelParallel() = default;
    ~ModelParallel();

    size_t num_partitions_ = 0;
    bool configured_ = false;

    struct Partition {
        MTL::CommandQueue* queue = nullptr;
    };

    std::vector<Partition> partitions_;
    MTL::SharedEvent* sync_event_ = nullptr;
    uint64_t event_value_ = 0;

    mutable std::mutex mutex_;
};

} // namespace cuda_metal
