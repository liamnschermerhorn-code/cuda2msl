cmake_minimum_required(VERSION 3.20)
project(cuda-to-metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES arm64)

# Metal-cpp headers
set(METAL_CPP_DIR ${CMAKE_SOURCE_DIR}/third_party/metal-cpp)

# macOS frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# Metal shader compilation support
include(cmake/MetalShaderSupport.cmake)

# Public include directory (drop-in CUDA headers)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${METAL_CPP_DIR})

add_subdirectory(src/shaders)
add_subdirectory(src/runtime)
add_subdirectory(src/cublas)
add_subdirectory(src/cusolver)
add_subdirectory(src/cusparse)
add_subdirectory(src/cufft)
add_subdirectory(src/curand)
add_subdirectory(src/cudnn)
add_subdirectory(src/nccl)
add_subdirectory(src/flash_attention)
add_subdirectory(src/aten_backend)
if(EXISTS ${CMAKE_SOURCE_DIR}/src/triton_backend/compiler.cpp AND
   EXISTS ${CMAKE_SOURCE_DIR}/src/triton_backend/launcher.cpp AND
   EXISTS ${CMAKE_SOURCE_DIR}/src/triton_backend/builtins.cpp)
    add_subdirectory(src/triton_backend)
endif()
add_subdirectory(src/nosana)
add_subdirectory(src/launcher)
add_subdirectory(src/transpiler)
add_subdirectory(src/inference)

# torch.distributed backend (depends on nccl_metal)
if(EXISTS ${CMAKE_SOURCE_DIR}/src/distributed/CMakeLists.txt)
    add_subdirectory(src/distributed)
endif()

# Export compile commands for clangd / IDE tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional: compile PyTorch Metal shaders into metallibs
option(BUILD_PYTORCH_SHADERS "Compile PyTorch .metal shaders from PYTORCH_SHADER_DIR" OFF)
set(PYTORCH_SHADER_DIR "" CACHE PATH "Directory containing PyTorch .metal shaders")

if(BUILD_PYTORCH_SHADERS AND PYTORCH_SHADER_DIR)
    file(GLOB PYTORCH_METAL_SOURCES "${PYTORCH_SHADER_DIR}/*.metal")
    list(LENGTH PYTORCH_METAL_SOURCES PYTORCH_SHADER_COUNT)
    if(PYTORCH_SHADER_COUNT GREATER 0)
        message(STATUS "Found ${PYTORCH_SHADER_COUNT} PyTorch Metal shaders in ${PYTORCH_SHADER_DIR}")
        add_metal_library(pytorch_kernels SOURCES ${PYTORCH_METAL_SOURCES})
        add_dependencies(cudart_metal pytorch_kernels)
    else()
        message(WARNING "No .metal files found in ${PYTORCH_SHADER_DIR}")
    endif()
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
