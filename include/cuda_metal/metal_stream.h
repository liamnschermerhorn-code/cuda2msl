#pragma once

#include <Metal/Metal.hpp>
#include <mutex>
#include <vector>
#include <set>

#include "cuda.h"

namespace cuda_metal {

struct MetalStream {
    MTL::CommandQueue*  queue = nullptr;
    MTL::CommandBuffer* active_cmd_buffer = nullptr;
    std::mutex          mutex;

    // Get or create the active command buffer
    MTL::CommandBuffer* getCommandBuffer();

    // Commit the active buffer and wait for completion
    void synchronize();

    // Commit the active buffer without waiting
    void flush();

    // Check if all work is complete
    bool isComplete();
};

struct MetalEvent {
    MTL::SharedEvent* shared_event = nullptr;
    uint64_t          signal_value = 0;
    double            gpu_start_time = 0.0;
    double            gpu_end_time = 0.0;
    bool              recorded = false;
    bool              timing_disabled = false;
};

class StreamManager {
public:
    static StreamManager& instance();

    cudaError_t createStream(cudaStream_t* pStream, unsigned int flags);
    cudaError_t destroyStream(cudaStream_t stream);
    cudaError_t synchronizeStream(cudaStream_t stream);
    cudaError_t queryStream(cudaStream_t stream);
    cudaError_t synchronizeAll();

    // Resolve nullptr to default stream
    MetalStream* resolve(cudaStream_t stream);

    // Events
    cudaError_t createEvent(cudaEvent_t* event, unsigned int flags);
    cudaError_t destroyEvent(cudaEvent_t event);
    cudaError_t recordEvent(cudaEvent_t event, cudaStream_t stream);
    cudaError_t synchronizeEvent(cudaEvent_t event);
    cudaError_t queryEvent(cudaEvent_t event);
    cudaError_t elapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop);

    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;

private:
    StreamManager();
    ~StreamManager();

    MetalStream default_stream_;
    std::set<MetalStream*> streams_;
    std::set<MetalEvent*> events_;
    std::mutex mutex_;
};

} // namespace cuda_metal
