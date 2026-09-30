#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace cuda_metal {

// ── MemoryPool — bump-allocator backed by a single Metal buffer ─────────
//
// Pre-allocates one large Metal buffer at init time. Subsequent alloc()
// calls return pointers within that buffer via a bump pointer (O(1)).
// reset() reclaims all allocations instantly (O(1)).
//
// This eliminates per-operation cudaMalloc/cudaFree overhead for scratch
// buffers in the forward pass.

class MemoryPool {
public:
    static MemoryPool& instance();

    // Initialize with a pre-allocated Metal buffer of the given size.
    // Must be called before any alloc(). Safe to call multiple times
    // (re-initializes if new size is larger).
    bool init(size_t pool_size);

    // Allocate from the pool. Returns nullptr if pool exhausted.
    // Alignment must be power of 2.
    void* alloc(size_t size, size_t alignment = 16);

    // Reset all allocations (O(1) — just resets bump pointer).
    void reset();

    // Check if a pointer belongs to this pool.
    bool contains(const void* ptr) const;

    // Get Metal buffer and offset for a pool pointer.
    // Returns {nullptr, 0} if ptr is not in the pool.
    std::pair<MTL::Buffer*, size_t> bufferAndOffset(const void* ptr) const;

    // Total pool size and current usage.
    size_t capacity() const { return pool_size_; }
    size_t used() const { return offset_; }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

private:
    MemoryPool() = default;
    ~MemoryPool();

    MTL::Buffer* pool_buffer_ = nullptr;
    uint8_t*     pool_base_   = nullptr;
    size_t       pool_size_   = 0;
    size_t       offset_      = 0;
};

} // namespace cuda_metal
