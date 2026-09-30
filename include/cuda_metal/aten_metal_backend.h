#pragma once

#include <cstddef>

namespace cuda_metal {

// ---------------------------------------------------------------------------
// MetalAllocator - wraps MetalMemoryManager (cudaMalloc/cudaFree) as a
// PyTorch-style allocator interface suitable for PrivateUse1 backend
// registration.
// ---------------------------------------------------------------------------
class MetalAllocator {
public:
    static MetalAllocator& instance();

    /// Allocate device memory through the Metal memory manager.
    /// Returns a host-accessible pointer into a MTL::Buffer (unified memory).
    void* allocate(size_t n);

    /// Free a previously allocated device pointer.
    void deallocate(void* ptr);

    MetalAllocator(const MetalAllocator&) = delete;
    MetalAllocator& operator=(const MetalAllocator&) = delete;

private:
    MetalAllocator() = default;
};

// ---------------------------------------------------------------------------
// MetalDeviceGuard - RAII device/stream context guard.
//
// Apple Silicon has a single GPU, so device switching is a no-op.  The guard
// exists to satisfy the PyTorch DeviceGuard protocol that backends must
// implement; it saves/restores the "current device index" thread-local.
// ---------------------------------------------------------------------------
class MetalDeviceGuard {
public:
    explicit MetalDeviceGuard(int device_index = 0);
    ~MetalDeviceGuard();

    MetalDeviceGuard(const MetalDeviceGuard&) = delete;
    MetalDeviceGuard& operator=(const MetalDeviceGuard&) = delete;

private:
    int prev_device_;
};

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

/// Initialize the Metal backend with PyTorch's PrivateUse1 dispatch key.
/// This registers the allocator, device guard factory, and all supported
/// aten operators (forward + backward).  Safe to call multiple times.
void initMetalBackend();

/// Query / set the "current" Metal device index (thread-local).
/// Always 0 on Apple Silicon, but the API must exist for the backend contract.
int  getCurrentMetalDevice();
void setCurrentMetalDevice(int device_index);

} // namespace cuda_metal

// ---------------------------------------------------------------------------
// C API for dlopen / Python ctypes usage
// ---------------------------------------------------------------------------
extern "C" {
void cudaMetalInitBackend();
}
