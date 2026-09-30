# Copy metallib to build root so tests can find shaders
add_custom_target(copy_metallib ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_BINARY_DIR}/src/shaders/compute_kernels.metallib
        ${CMAKE_CURRENT_BINARY_DIR}/compute_kernels.metallib
    DEPENDS compute_kernels
    COMMENT "Copying compute_kernels.metallib to test directory"
)

# Unit tests
add_executable(test_memory unit/test_memory.cpp)
target_link_libraries(test_memory cudart_metal)
add_test(NAME test_memory COMMAND test_memory WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_device unit/test_device.cpp)
target_link_libraries(test_device cudart_metal)
add_test(NAME test_device COMMAND test_device WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_stream unit/test_stream.cpp)
target_link_libraries(test_stream cudart_metal)
add_test(NAME test_stream COMMAND test_stream WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_event unit/test_event.cpp)
target_link_libraries(test_event cudart_metal)
add_test(NAME test_event COMMAND test_event WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_gemm unit/test_gemm.cpp)
target_link_libraries(test_gemm cudart_metal cublas_metal)
add_test(NAME test_gemm COMMAND test_gemm WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_activation unit/test_activation.cpp)
target_link_libraries(test_activation cudart_metal cudnn_metal)
add_test(NAME test_activation COMMAND test_activation WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(test_activation copy_metallib)

add_executable(test_softmax unit/test_softmax.cpp)
target_link_libraries(test_softmax cudart_metal cudnn_metal)
add_test(NAME test_softmax COMMAND test_softmax WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(test_softmax copy_metallib)

add_executable(test_secure_weight_pool unit/test_secure_weight_pool.cpp)
target_link_libraries(test_secure_weight_pool cudart_metal)
add_test(NAME test_secure_weight_pool COMMAND test_secure_weight_pool WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Nosana integration test
add_executable(test_nosana unit/test_nosana.cpp)
target_link_libraries(test_nosana cudart_metal nvidia_ml)
add_test(NAME test_nosana COMMAND test_nosana WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Integration tests — model-scale patterns
add_executable(test_model_patterns integration/test_model_patterns.cpp)
target_link_libraries(test_model_patterns cudart_metal cublas_metal cudnn_metal)
add_test(NAME test_model_patterns COMMAND test_model_patterns WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(test_model_patterns copy_metallib)

# Integration tests — large model patterns (GPT-2, LLaMA, ResNet-50)
add_executable(test_large_models integration/test_large_models.cpp)
target_link_libraries(test_large_models cudart_metal cublas_metal cudnn_metal)
add_test(NAME test_large_models COMMAND test_large_models WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(test_large_models copy_metallib)

# Integration tests — LLM inference patterns (Claude-scale)
add_executable(test_llm_inference integration/test_llm_inference.cpp)
target_link_libraries(test_llm_inference cudart_metal cublas_metal cudnn_metal)
add_test(NAME test_llm_inference COMMAND test_llm_inference WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(test_llm_inference copy_metallib)

# Integration tests — full Claude-architecture model
add_executable(test_claude_model integration/test_claude_model.cpp)
target_link_libraries(test_claude_model cudart_metal cublas_metal cudnn_metal)
add_test(NAME test_claude_model COMMAND test_claude_model WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(test_claude_model copy_metallib)

# Multi-model architecture tests (Mistral, Mixtral, DeepSeek, Whisper, BERT)
add_executable(test_multi_model integration/test_multi_model.cpp)
target_link_libraries(test_multi_model cudart_metal cublas_metal)
add_test(NAME test_multi_model COMMAND test_multi_model WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(test_multi_model copy_metallib)

# Benchmarks (not in ctest — run manually)
add_executable(bench_gemm benchmarks/bench_gemm.cpp)
target_link_libraries(bench_gemm cudart_metal cublas_metal)
