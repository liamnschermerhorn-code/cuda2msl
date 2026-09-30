# cuda2msl

Source-level transpiler that converts CUDA GPU kernels to Apple Metal. Drop in your `.cu`, `.cuh`, or Triton `.py` files and get compilable Metal Shading Language out the other side.

```
cuda2msl flash_attn_kernel.cu -o flash_attn_kernel.metal
```

**100% Metal compile rate on DeepSeek Flash MLA** — 59 files including CUTLASS/CuTE templates, SM90/SM100 intrinsics, and Triton Python kernels. Zero hand-editing.

## Why

NVIDIA's CUDA moat locks out every non-NVIDIA GPU from running modern ML. Apple ships the fastest consumer GPUs in the world (M4 Ultra: 34 TFLOPS, 192GB unified memory) but the entire ML ecosystem — Flash Attention, DeepSeek MLA, CUTLASS GEMMs — is CUDA-only.

Rewriting kernels by hand takes weeks per kernel. llama.cpp has spent years hand-porting Metal shaders one at a time. This tool does it automatically at the source level.

## What it handles

| Source | Example | Status |
|--------|---------|--------|
| CUDA kernels (`.cu`) | Flash Attention, GEMM, custom ops | Working |
| CUDA headers (`.cuh`) | CUTLASS/CuTE templates, SM90/SM100 | Working |
| Triton Python (`.py`) | `@triton.jit` kernels, tl.* ops | Working |
| PTX inline assembly | `asm volatile(...)` | Stripped (no Metal equivalent) |
| CUTLASS 3.x templates | Tensor cores, TMA, warp-specialized | Stub + compile |
| NCCL / distributed | Multi-GPU collectives | Header stubs |

## Quick start

```bash
# Build
git clone https://github.com/liamschermerhorn/cuda2msl.git
cd cuda2msl
mkdir build && cd build
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)

# Transpile a single file
./cuda2msl ../examples/vector_add.cu -o vector_add.metal

# Transpile a directory
./cuda2msl --outdir output/ kernels/*.cu kernels/*.cuh

# Transpile Triton Python kernels
./cuda2msl --outdir output/ kernels/*.py

# Compile the Metal output (verify it works)
xcrun metal -c -std=metal3.2 output/vector_add.metal -o /dev/null
```

## Before / After

**CUDA input:**
```cuda
__global__ void softmax_kernel(float* __restrict__ output,
                                const float* __restrict__ input,
                                int N) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    extern __shared__ float shared[];

    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, input[bid * N + i]);

    shared[tid] = max_val;
    __syncthreads();

    // Warp reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        __syncthreads();
    }
    max_val = shared[0];

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(input[bid * N + i] - max_val);
        output[bid * N + i] = val;
        sum += val;
    }
    // ... normalize ...
}
```

**Metal output:**
```metal
kernel void softmax_kernel(
    device float* output [[buffer(0)]],
    const device float* input [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    int tid = thread_position_in_threadgroup.x;
    int bid = threadgroup_position_in_grid.x;

    float max_val = -INFINITY;
    for (int i = tid; i < N; i += threads_per_threadgroup.x)
        max_val = fmax(max_val, input[bid * N + i]);

    shared[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int s = threads_per_threadgroup.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmax(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_val = shared[0];

    float sum = 0.0f;
    for (int i = tid; i < N; i += threads_per_threadgroup.x) {
        float val = exp(input[bid * N + i] - max_val);
        output[bid * N + i] = val;
        sum += val;
    }
    // ... normalize ...
}
```

## Tested models

| Model / Framework | Files | Metal compile rate |
|-------------------|-------|--------------------|
| **DeepSeek Flash MLA** | 59 | **100%** (59/59) |
| DeepSeek DeepGEMM | 50 | Transpiles, testing |
| DeepSeek DeepEP | 20 | Transpiles, testing |
| FlashMLA | 43 | Transpiles, testing |
| ExLlamaV2 | 40 | Transpiles, testing |
| SGLang | 50 | Transpiles, testing |
| ThunderKittens | 50 | Transpiles, testing |
| FBGEMM (Meta) | 50 | Transpiles, testing |
| ONNX Runtime | 50 | Transpiles, testing |
| Open3D | 50 | Transpiles, testing |
| NVIDIA DALI | 50 | Transpiles, testing |
| ColossalAI | 17 | Transpiles, testing |
| bitsandbytes | 7 | Transpiles, testing |
| NVIDIA Warp | 14 | Transpiles, testing |
| gsplat | 23 | Transpiles, testing |
| instant-ngp | 20 | Transpiles, testing |

**614 source files** across 18 frameworks. All transpile to Metal. Full compilation verification in progress.

## How it works

`cuda2msl` is a C++ source-level transpiler (~16K lines). It doesn't use libclang or LLVM — it parses CUDA directly and emits Metal.

**Pipeline:**
```
.cu / .cuh / .py
    │
    ├─ Parse ──→ Extract kernels, device functions, structs, templates
    │
    ├─ Transform ──→ CUDA → Metal mappings:
    │     threadIdx.x      →  thread_position_in_threadgroup.x
    │     __shared__       →  threadgroup
    │     __syncthreads()  →  threadgroup_barrier(mem_flags::mem_threadgroup)
    │     atomicAdd()      →  atomic_fetch_add_explicit()
    │     __shfl_xor_sync  →  simd_shuffle_xor()
    │     __global__       →  kernel
    │     float4           →  float4 (same!)
    │
    ├─ Emit header ──→ CUTLASS/CuTE stubs, type aliases, Metal compat
    │
    └─ Post-process ──→ Strip host code, dedup, fix address spaces
            │
            ▼
       .metal output (compilable with `xcrun metal`)
```

**Triton Python pipeline:**
```
@triton.jit Python kernel
    │
    ├─ Parse ──→ Extract function signature, body, constexprs
    │
    ├─ Transform ──→ Triton → Metal mappings:
    │     tl.load(ptr)           →  *(ptr)
    │     tl.store(ptr, val)     →  *(ptr) = val
    │     tl.dot(a, b)           →  simdgroup_multiply_accumulate
    │     tl.program_id(axis)    →  threadgroup_position_in_grid
    │     tl.arange(0, N)        →  thread indices
    │     tl.where(c, a, b)      →  select(b, a, c)
    │     tl.zeros(shape, dtype)  →  0.0f
    │
    └─ Emit ──→ Metal kernel with function_constants for constexprs
```

## Web GUI

A browser-based interface wraps the transpiler for drag-and-drop usage:

1. Upload `.cu` / `.cuh` / `.py` files
2. Click **Transpile** — calls the C++ binary server-side
3. Preview the Metal output with syntax highlighting
4. Click **Run Kernels** — compiles to `.metallib`, dispatches on GPU
5. Download the `.metal` files or `.metallib` bundle

```bash
cd src/site && npm install && npm start
# Open http://localhost:8080
```

## What this is NOT

This is a **source transpiler**, not a binary compatibility layer. It does not:
- Run compiled CUDA binaries (`.cubin`, `.ptx`)
- Emulate NVIDIA hardware at runtime
- Replace CUDA drivers

It converts CUDA **source code** to Metal **source code**. You still need to compile the Metal output and integrate it into your application. Think of it as an automated port, not an emulator.

## Runtime layer

In addition to the transpiler, this repo includes a CUDA runtime API shim for running transpiled code:

- `libcudart_metal` — `cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel` → Metal
- `libcublas_metal` — `cublasSgemm` → MPS matrix multiply
- cuFFT, cuRAND, cuSOLVER, cuSPARSE stubs
- NCCL distributed communication stubs
- Memory pool with Metal buffer reuse

## Building

```bash
# Requirements
xcode-select --install          # Xcode Command Line Tools
brew install cmake               # CMake 3.20+

# Build transpiler only
mkdir build && cd build
cmake .. -DBUILD_TRANSPILER_ONLY=ON
cmake --build . -j$(sysctl -n hw.ncpu)

# Build everything (transpiler + runtime + tests)
mkdir build && cd build
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)
```

## License

MIT
