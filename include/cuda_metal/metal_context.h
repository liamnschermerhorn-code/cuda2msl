#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <mutex>
#include <string>

namespace cuda_metal {

class MetalContext {
public:
    static MetalContext& instance();

    MTL::Device*       device();
    MTL::CommandQueue* defaultQueue();
    MTL::Library*      shaderLibrary();

    // Device info
    const std::string& deviceName() const;
    size_t             maxBufferLength() const;
    size_t             recommendedMaxWorkingSetSize() const;

    // Prevent copies
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

private:
    MetalContext();
    ~MetalContext();

    MTL::Device*       device_;
    MTL::CommandQueue* default_queue_;
    MTL::Library*      shader_library_;  // may be null if metallib not found
    std::string        device_name_;
};

} // namespace cuda_metal
