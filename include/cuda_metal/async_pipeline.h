#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <mutex>
#include <vector>

namespace cuda_metal {

// Triple-buffered command buffer pipeline for overlapping GPU work.
//
// Design:
//  - N command buffer slots (default 3) rotate in ring-buffer fashion
//  - currentCommandBuffer() returns the slot being encoded
//  - advance() commits current, rotates to next (blocks if next is still in-flight)
//  - drain() waits for all in-flight work
//  - Phases (compute/memory) hint the encoder to overlap where hardware supports it

class AsyncPipeline {
public:
    static AsyncPipeline& instance();

    // Configure the pipeline. Creates N command buffer slots.
    // Must be called after MetalContext is initialized.
    bool configure(size_t num_buffers = 3);

    // Get the current command buffer for encoding.
    MTL::CommandBuffer* currentCommandBuffer();

    // Commit current command buffer and rotate to next slot.
    // Blocks if the next slot hasn't completed yet.
    void advance();

    // Begin a compute encoding phase on the current command buffer.
    MTL::ComputeCommandEncoder* beginComputePhase();

    // End the current compute phase.
    void endComputePhase();

    // Begin a blit (memory copy) phase on the current command buffer.
    MTL::BlitCommandEncoder* beginMemoryPhase();

    // End the current memory phase.
    void endMemoryPhase();

    // Wait for all in-flight command buffers to complete.
    void drain();

    // Number of configured buffer slots.
    size_t numBuffers() const { return num_buffers_; }

    // Whether the pipeline has been configured.
    bool isConfigured() const { return configured_; }

    AsyncPipeline(const AsyncPipeline&) = delete;
    AsyncPipeline& operator=(const AsyncPipeline&) = delete;

private:
    AsyncPipeline() = default;
    ~AsyncPipeline();

    size_t num_buffers_ = 0;
    size_t current_    = 0;
    bool   configured_ = false;

    struct Slot {
        MTL::CommandBuffer*         cmd_buffer = nullptr;
        MTL::ComputeCommandEncoder* compute_encoder = nullptr;
        MTL::BlitCommandEncoder*    blit_encoder = nullptr;
        bool                        in_flight = false;
    };

    std::vector<Slot> slots_;
    MTL::CommandQueue* queue_ = nullptr;
    mutable std::mutex mutex_;

    // Allocate a fresh command buffer for the given slot.
    void prepareSlot(size_t index);
};

} // namespace cuda_metal
