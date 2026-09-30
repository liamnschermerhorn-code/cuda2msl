#pragma once

#include <Metal/Metal.hpp>
#include <map>
#include <mutex>
#include <set>
#include <utility>

#include "cuda.h"

namespace cuda_metal {

class MetalMemoryManager {
public:
    static MetalMemoryManager& instance();

    // Allocate a shared MTL::Buffer, return buffer->contents() as device pointer
    cudaError_t allocate(void** ptr, size_t size);

    // Free a previously allocated device pointer
    cudaError_t free(void* ptr);

    // Lookup the MTL::Buffer containing the given pointer
    // Returns nullptr if not found
    MTL::Buffer* findBuffer(const void* ptr);

    // Lookup buffer and compute offset from buffer start
    // Returns {nullptr, 0} if not found
    std::pair<MTL::Buffer*, size_t> findBufferAndOffset(const void* ptr);

    // Register externally-owned memory (e.g., mmap'd) as a Metal buffer.
    // The pointer must be page-aligned. The caller retains ownership of the memory.
    // The Metal buffer wraps the memory without copying (zero-copy).
    cudaError_t registerExternal(void* ptr, size_t size);

    // Unregister externally-owned memory (releases Metal buffer, not the memory).
    cudaError_t unregisterExternal(void* ptr);

    // Check if a pointer is a known device allocation
    bool isDevicePointer(const void* ptr) const;

    MetalMemoryManager(const MetalMemoryManager&) = delete;
    MetalMemoryManager& operator=(const MetalMemoryManager&) = delete;

private:
    MetalMemoryManager() = default;
    ~MetalMemoryManager();

    // Ordered map: base_ptr -> {MTL::Buffer*, size}
    // Using ordered map enables lower_bound lookup for sub-allocation pointers
    std::map<uintptr_t, std::pair<MTL::Buffer*, size_t>> allocations_;
    std::set<uintptr_t> external_; // externally-owned (don't free memory on release)
    mutable std::mutex mutex_;
};

} // namespace cuda_metal
