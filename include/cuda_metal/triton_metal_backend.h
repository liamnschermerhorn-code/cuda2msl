#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace cuda_metal { namespace triton {

// Represents a compiled Triton kernel for Metal
struct CompiledKernel {
    std::string name;
    std::string msl_source;     // Generated MSL code
    void* pipeline_state;       // MTL::ComputePipelineState* (opaque)
    int num_warps;
    int num_stages;
    int shared_memory_bytes;
};

// Triton operation types (subset of Triton IR)
enum class TritonOp {
    Load, Store, Dot, Reduce,
    Add, Sub, Mul, Div,
    Exp, Log, Sqrt, Abs,
    Maximum, Minimum,
    CmpEq, CmpLt, CmpGt,
    Select, Cast,
    Broadcast, Splat, Reshape,
    ProgramId, MakeRange, ExpandDims
};

// IR node for the Triton Metal compiler
struct TritonIRNode {
    TritonOp op;
    std::string result_name;
    std::vector<std::string> operands;
    std::unordered_map<std::string, std::string> attrs;  // e.g., {"axis": "0", "dtype": "f32"}
};

// A Triton kernel represented as IR
struct TritonKernelIR {
    std::string name;
    std::vector<std::string> arg_names;
    std::vector<std::string> arg_types;  // "ptr_f32", "i32", "f32", etc.
    std::vector<std::string> constexpr_names;
    std::vector<int> constexpr_values;  // e.g., BLOCK_SIZE
    std::vector<TritonIRNode> body;
    int num_warps = 4;
    int num_stages = 2;
};

// Main compiler class
class TritonMetalCompiler {
public:
    static TritonMetalCompiler& instance();

    // Compile Triton IR to MSL
    std::string compileToMSL(const TritonKernelIR& kernel);

    // Compile and cache kernel
    CompiledKernel* compileKernel(const TritonKernelIR& kernel);

    // Launch a compiled kernel
    cudaError_t launchKernel(const CompiledKernel* kernel,
                              uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                              void** args, size_t num_args,
                              cudaStream_t stream = nullptr);

    // Get cached kernel by name
    CompiledKernel* getCachedKernel(const std::string& name);

private:
    TritonMetalCompiler() = default;
    std::unordered_map<std::string, CompiledKernel> cache_;

    // MSL code generation helpers
    std::string generateHeader(const TritonKernelIR& kernel);
    std::string generateBody(const TritonKernelIR& kernel);
    std::string mapOp(const TritonIRNode& node);
    std::string mapType(const std::string& triton_type);
};

// Pre-built common kernels (vector add, softmax, matmul, etc.)
void registerBuiltinTritonKernels();

}} // namespace cuda_metal::triton
