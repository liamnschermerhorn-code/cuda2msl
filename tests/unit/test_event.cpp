#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

#define CHECK(call)                                                         \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "FAIL: %s returned %d (%s) at %s:%d\n",        \
                    #call, err, cudaGetErrorString(err), __FILE__, __LINE__);\
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static void test_event_create_destroy() {
    printf("  test_event_create_destroy... ");
    cudaEvent_t event;
    CHECK(cudaEventCreate(&event));
    assert(event != nullptr);
    CHECK(cudaEventDestroy(event));
    printf("PASS\n");
}

static void test_event_with_flags() {
    printf("  test_event_with_flags... ");
    cudaEvent_t event;
    CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    assert(event != nullptr);
    CHECK(cudaEventDestroy(event));
    printf("PASS\n");
}

static void test_event_record_sync() {
    printf("  test_event_record_sync... ");
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cudaEvent_t event;
    CHECK(cudaEventCreate(&event));

    // Do some work on the stream
    void* dev;
    CHECK(cudaMalloc(&dev, 1024));
    CHECK(cudaMemset(dev, 0, 1024));

    CHECK(cudaEventRecord(event, stream));
    CHECK(cudaEventSynchronize(event));

    // Query should succeed after sync
    cudaError_t status = cudaEventQuery(event);
    assert(status == cudaSuccess);

    CHECK(cudaFree(dev));
    CHECK(cudaEventDestroy(event));
    CHECK(cudaStreamDestroy(stream));
    printf("PASS\n");
}

static void test_event_elapsed_time() {
    printf("  test_event_elapsed_time... ");
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start, nullptr));

    // Do some work
    void* dev;
    CHECK(cudaMalloc(&dev, 1024 * 1024));
    CHECK(cudaMemset(dev, 0, 1024 * 1024));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(stop, nullptr));
    CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = -1.0f;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Elapsed time should be non-negative
    printf("(%.3f ms) ", elapsed_ms);
    assert(elapsed_ms >= 0.0f);

    CHECK(cudaFree(dev));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    printf("PASS\n");
}

int main() {
    printf("=== Event Tests ===\n");
    test_event_create_destroy();
    test_event_with_flags();
    test_event_record_sync();
    test_event_elapsed_time();
    printf("All event tests PASSED\n");
    return 0;
}
