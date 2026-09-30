#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "FAIL: %s returned %d (%s) at %s:%d\n",        \
                    #call, err, cudaGetErrorString(err), __FILE__, __LINE__);\
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static void test_device_count() {
    printf("  test_device_count... ");
    int count = -1;
    CHECK(cudaGetDeviceCount(&count));
    assert(count == 1);
    printf("PASS\n");
}

static void test_get_set_device() {
    printf("  test_get_set_device... ");
    int device = -1;
    CHECK(cudaGetDevice(&device));
    assert(device == 0);

    CHECK(cudaSetDevice(0));

    // Setting invalid device should fail
    cudaError_t err = cudaSetDevice(1);
    assert(err == cudaErrorInvalidDevice);
    printf("PASS\n");
}

static void test_device_properties() {
    printf("  test_device_properties... ");
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    CHECK(cudaGetDeviceProperties(&prop, 0));

    // Name should be non-empty
    assert(strlen(prop.name) > 0);
    printf("    Device: %s\n", prop.name);

    // Memory should be > 0
    assert(prop.totalGlobalMem > 0);
    printf("    Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));

    // Compute capability should be >= 8.0
    assert(prop.major >= 8);

    // Warp size should be 32
    assert(prop.warpSize == 32);

    // Max threads per block should be 1024
    assert(prop.maxThreadsPerBlock == 1024);

    // Shared memory should be > 0
    assert(prop.sharedMemPerBlock > 0);

    // Unified addressing
    assert(prop.unifiedAddressing == 1);

    printf("  PASS\n");
}

static void test_error_strings() {
    printf("  test_error_strings... ");
    assert(strcmp(cudaGetErrorString(cudaSuccess), "no error") == 0);
    assert(strcmp(cudaGetErrorName(cudaSuccess), "cudaSuccess") == 0);

    assert(strlen(cudaGetErrorString(cudaErrorMemoryAllocation)) > 0);
    assert(strlen(cudaGetErrorName(cudaErrorMemoryAllocation)) > 0);
    printf("PASS\n");
}

static void test_last_error() {
    printf("  test_last_error... ");
    // Clear any previous error
    cudaGetLastError();

    // Trigger an error
    cudaSetDevice(99);

    // Peek should see the error but not clear it
    cudaError_t err = cudaPeekAtLastError();
    assert(err == cudaErrorInvalidDevice);

    // GetLastError should clear it
    err = cudaGetLastError();
    assert(err == cudaErrorInvalidDevice);

    // Now it should be clear
    err = cudaGetLastError();
    assert(err == cudaSuccess);
    printf("PASS\n");
}

int main() {
    printf("=== Device Tests ===\n");
    test_device_count();
    test_get_set_device();
    test_device_properties();
    test_error_strings();
    test_last_error();
    printf("All device tests PASSED\n");
    return 0;
}
