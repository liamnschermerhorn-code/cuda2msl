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

static void test_malloc_free() {
    printf("  test_malloc_free... ");
    void* ptr = nullptr;
    CHECK(cudaMalloc(&ptr, 1024));
    assert(ptr != nullptr);
    CHECK(cudaFree(ptr));

    // Free nullptr should be no-op
    CHECK(cudaFree(nullptr));

    // Zero-size allocation
    void* ptr2 = (void*)0xDEAD;
    CHECK(cudaMalloc(&ptr2, 0));
    assert(ptr2 == nullptr);
    printf("PASS\n");
}

static void test_memcpy_roundtrip() {
    printf("  test_memcpy_roundtrip... ");
    const int N = 256;
    float host_src[N], host_dst[N];
    for (int i = 0; i < N; i++) host_src[i] = (float)i;
    memset(host_dst, 0, sizeof(host_dst));

    void* dev;
    CHECK(cudaMalloc(&dev, N * sizeof(float)));
    CHECK(cudaMemcpy(dev, host_src, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(host_dst, dev, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dev));

    for (int i = 0; i < N; i++) {
        assert(host_dst[i] == (float)i);
    }
    printf("PASS\n");
}

static void test_memcpy_device_to_device() {
    printf("  test_memcpy_device_to_device... ");
    const int N = 128;
    float host_src[N], host_dst[N];
    for (int i = 0; i < N; i++) host_src[i] = (float)(i * 2);

    void* dev1;
    void* dev2;
    CHECK(cudaMalloc(&dev1, N * sizeof(float)));
    CHECK(cudaMalloc(&dev2, N * sizeof(float)));

    CHECK(cudaMemcpy(dev1, host_src, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev2, dev1, N * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(host_dst, dev2, N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dev1));
    CHECK(cudaFree(dev2));

    for (int i = 0; i < N; i++) {
        assert(host_dst[i] == (float)(i * 2));
    }
    printf("PASS\n");
}

static void test_memset() {
    printf("  test_memset... ");
    const int N = 256;
    void* dev;
    CHECK(cudaMalloc(&dev, N));
    CHECK(cudaMemset(dev, 0, N));

    unsigned char host[N];
    CHECK(cudaMemcpy(host, dev, N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        assert(host[i] == 0);
    }

    CHECK(cudaMemset(dev, 0xFF, N));
    CHECK(cudaMemcpy(host, dev, N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        assert(host[i] == 0xFF);
    }

    CHECK(cudaFree(dev));
    printf("PASS\n");
}

static void test_malloc_host() {
    printf("  test_malloc_host... ");
    void* ptr = nullptr;
    CHECK(cudaMallocHost(&ptr, 4096));
    assert(ptr != nullptr);

    // Should be usable as regular memory
    memset(ptr, 42, 4096);
    assert(((unsigned char*)ptr)[0] == 42);

    CHECK(cudaFreeHost(ptr));
    printf("PASS\n");
}

static void test_large_allocation() {
    printf("  test_large_allocation... ");
    // Allocate 64 MB
    size_t size = 64 * 1024 * 1024;
    void* dev;
    CHECK(cudaMalloc(&dev, size));
    assert(dev != nullptr);
    CHECK(cudaMemset(dev, 0, size));
    CHECK(cudaFree(dev));
    printf("PASS\n");
}

int main() {
    printf("=== Memory Tests ===\n");
    test_malloc_free();
    test_memcpy_roundtrip();
    test_memcpy_device_to_device();
    test_memset();
    test_malloc_host();
    test_large_allocation();
    printf("All memory tests PASSED\n");
    return 0;
}
