#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace cuda_metal {

// Opaque handle to a securely managed weight buffer
struct SecureWeightHandle {
    uint64_t id;           // unique allocation identifier
    void*    host_ptr;     // CPU-accessible pointer (zero-copy via StorageModeShared)
    size_t   weight_bytes; // payload size (excludes guard regions)
};

// Integrity violation callback signature
using IntegrityCallback = std::function<void(uint64_t handle_id, const char* reason)>;

// SecureWeightPool
//
// MPS-aware weight buffer manager with:
//   - Zero-copy access via Metal unified memory (StorageModeShared)
//   - Cryptographic zeroization on release (Strict Isolation)
//   - Canary-guarded buffer boundaries
//   - Background integrity watchdog (Kernel Heartbeat)
//
// Designed for large model weights (DeepSeek-V3 scale: 671B MoE, ~100+ GB).
// All Metal/MPS types are hidden behind void* so this header is pure C++.
class SecureWeightPool {
public:
    static SecureWeightPool& instance();

    // ── Allocation ─────────────────────────────────────────────────────

    // Allocate a secure weight buffer.  The returned handle's host_ptr is
    // directly usable by both CPU and GPU (zero-copy on Apple Silicon).
    SecureWeightHandle allocate(size_t weight_bytes);

    // Load weights from a file via mmap into a secure buffer.
    // Returns handle with host_ptr pointing to the mapped data.
    // The file is memory-mapped read-only; contents are copied into a
    // Metal shared buffer so the mapping can be released immediately.
    SecureWeightHandle load_from_file(const char* path, size_t offset, size_t length);

    // ── Access ─────────────────────────────────────────────────────────

    // Get the underlying Metal buffer (MTL::Buffer*) as void* for MPS dispatch.
    void* get_metal_buffer(uint64_t handle_id);

    // Get the host-side pointer (same as handle.host_ptr, but validated).
    void* get_host_ptr(uint64_t handle_id);

    // Get payload size for a handle.
    size_t get_weight_bytes(uint64_t handle_id);

    // ── Release (Strict Isolation) ─────────────────────────────────────

    // Scrub the buffer contents with cryptographic zeroization, verify guard
    // canaries, then release the Metal buffer back to the system.
    void release(uint64_t handle_id);

    // Release all managed buffers with full scrub.
    void release_all();

    // ── Integrity Watchdog (Kernel Heartbeat) ──────────────────────────

    // Start the background heartbeat thread.  check_interval_ms controls
    // how often canary + pointer-consistency checks run.
    // violation_cb is invoked on the watchdog thread if a violation is found.
    void start_heartbeat(uint32_t check_interval_ms, IntegrityCallback violation_cb);

    // Stop the heartbeat thread.
    void stop_heartbeat();

    // Run a single integrity sweep (callable from any thread).
    // Returns true if all buffers pass, false if a violation was found.
    bool check_integrity();

    // ── Stats ──────────────────────────────────────────────────────────

    size_t   active_count() const;
    size_t   total_bytes_managed() const;

    SecureWeightPool(const SecureWeightPool&) = delete;
    SecureWeightPool& operator=(const SecureWeightPool&) = delete;

private:
    SecureWeightPool();
    ~SecureWeightPool();

    struct Impl;
    Impl* impl_;
};

} // namespace cuda_metal
