#pragma once

#include <Metal/Metal.hpp>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cuda_metal {

// Function constant value (bool, int, float)
using FunctionConstant = std::variant<bool, int32_t, uint32_t, float>;

// A set of function constant values keyed by index
using FunctionConstants = std::unordered_map<uint32_t, FunctionConstant>;

/// ShaderLoader manages the lifecycle of Metal shader libraries and
/// auto-registers all kernel functions for dispatch via cudaLaunchKernel.
///
/// Usage:
///   ShaderLoader::instance().loadMetalLib("pytorch_kernels.metallib");
///   ShaderLoader::instance().loadDirectory("/path/to/shaders/");
///   auto* pipeline = ShaderLoader::instance().findKernel("elementwise_kernel");
class ShaderLoader {
public:
    static ShaderLoader& instance();

    /// Load a pre-compiled .metallib file and register all kernels
    bool loadMetalLib(const std::string& path);

    /// Load an encrypted .metallib (M5+ obfuscated shaders)
    bool loadProtectedMetalLib(const std::string& path,
                                const std::string& passphrase = "");

    /// Load a .metallib from raw bytes in memory (for decrypted data)
    bool loadMetalLibFromData(const void* data, size_t size,
                               const std::string& label = "");

    /// Compile Metal source code at runtime and register all kernels
    bool compileAndLoadSource(const std::string& source, const std::string& name = "");

    /// Load all .metallib and .metal files from a directory
    bool loadDirectory(const std::string& dir_path);

    /// Look up a registered kernel pipeline by name
    MTL::ComputePipelineState* findKernel(const std::string& name) const;

    /// Create a specialized kernel pipeline with function constants (M5+).
    /// The specialized pipeline is cached by name + constants hash.
    MTL::ComputePipelineState* findOrCreateSpecialized(
        const std::string& kernel_name,
        const FunctionConstants& constants);

    /// List all registered kernel names
    std::vector<std::string> listKernels() const;

    /// Number of registered kernels
    size_t kernelCount() const;

    ShaderLoader(const ShaderLoader&) = delete;
    ShaderLoader& operator=(const ShaderLoader&) = delete;

private:
    ShaderLoader() = default;
    ~ShaderLoader();

    void registerAllKernels(MTL::Library* lib);

    // Build a MTL::FunctionConstantValues from our portable map
    MTL::FunctionConstantValues* buildConstantValues(
        const FunctionConstants& constants) const;

    // Hash function constants for cache key
    static size_t hashConstants(const FunctionConstants& c);

    std::unordered_map<std::string, MTL::ComputePipelineState*> kernel_pipelines_;
    std::unordered_map<std::string, MTL::ComputePipelineState*> specialized_pipelines_;
    std::vector<MTL::Library*> libraries_;
    mutable std::mutex mutex_;
};

} // namespace cuda_metal
