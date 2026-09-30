#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cuda_metal {

// Size-class bucketed memory pool with individual free() support.
// Replaces the bump-only allocator for production inference.
//
// Design:
//  - 7 size classes: 256B, 1K, 4K, 16K, 64K, 256K, 1M
//  - Each class maintains a free list of pre-split blocks
//  - Large allocations (>1M) get dedicated Metal buffers
//  - Thread-safe via mutex (GPU dispatch is serialized anyway)
//  - Compatible with existing MemoryPool::reset() pattern

struct PoolStats {
    size_t total_bytes    = 0;  // Total memory managed
    size_t used_bytes     = 0;  // Currently allocated
    size_t peak_bytes     = 0;  // High-water mark
    size_t alloc_count    = 0;  // Number of alloc() calls
    size_t free_count     = 0;  // Number of free() calls
    size_t cache_hits     = 0;  // Free-list reuse
    size_t cache_misses   = 0;  // New block created
};

class MemoryPoolV2 {
public:
    static MemoryPoolV2& instance();

    // Initialize the pool. Creates the backing Metal buffers.
    // safe to call multiple times.
    bool init(size_t initial_size = 64 * 1024 * 1024);

    // Allocate memory. Returns a unified (CPU+GPU) pointer.
    void* alloc(size_t size, size_t alignment = 16);

    // Free a previously allocated pointer. Returns it to the free list.
    void free(void* ptr);

    // Reset all allocations (backward-compatible with bump allocator).
    void reset();

    // Get the Metal buffer and offset for a pool pointer.
    std::pair<MTL::Buffer*, size_t> bufferAndOffset(const void* ptr) const;

    // Check if a pointer belongs to this pool.
    bool contains(const void* ptr) const;

    // Acquire/release a Metal buffer directly (for kernel argument binding).
    MTL::Buffer* acquireBuffer(size_t size);
    void releaseBuffer(MTL::Buffer* buffer);

    // Statistics
    PoolStats stats() const;

    MemoryPoolV2(const MemoryPoolV2&) = delete;
    MemoryPoolV2& operator=(const MemoryPoolV2&) = delete;

private:
    MemoryPoolV2() = default;
    ~MemoryPoolV2();

    // Size classes (bytes)
    static constexpr size_t kNumClasses = 7;
    static constexpr size_t kClassSizes[kNumClasses] = {
        256, 1024, 4096, 16384, 65536, 262144, 1048576
    };

    // Find the smallest size class >= requested size. Returns kNumClasses for oversized.
    static size_t sizeClassFor(size_t size);

    struct Block {
        void*        ptr;
        size_t       size;
        MTL::Buffer* buffer;  // Owning Metal buffer
        size_t       offset;  // Offset within buffer
    };

    // Per-class free lists
    std::vector<Block> free_lists_[kNumClasses];

    // Oversized: dedicated buffers
    struct LargeAlloc {
        MTL::Buffer* buffer;
        size_t       size;
    };
    std::unordered_map<void*, LargeAlloc> large_allocs_;

    // Track all allocations for contains() / bufferAndOffset()
    std::unordered_map<void*, Block> active_allocs_;

    // Backing slab: one large Metal buffer split into blocks
    MTL::Buffer* slab_buffer_  = nullptr;
    uint8_t*     slab_base_    = nullptr;
    size_t       slab_size_    = 0;
    size_t       slab_offset_  = 0;  // Bump pointer for new block creation within slab

    // Metal buffer cache (for acquireBuffer/releaseBuffer)
    std::vector<MTL::Buffer*> buffer_cache_;

    mutable std::mutex mutex_;
    PoolStats stats_;
    bool initialized_ = false;

    // Allocate a new block from the slab or a new Metal buffer
    Block allocBlock(size_t size);
};

} // namespace cuda_metal
