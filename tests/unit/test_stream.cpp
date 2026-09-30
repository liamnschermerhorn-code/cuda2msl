#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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

static void test_stream_create_destroy() {
    printf("  test_stream_create_destroy... ");
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    assert(stream != nullptr);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaStreamDestroy(stream));
    printf("PASS\n");
}

static void test_stream_with_flags() {
    printf("  test_stream_with_flags... ");
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    assert(stream != nullptr);
    CHECK(cudaStreamDestroy(stream));
    printf("PASS\n");
}

static void test_default_stream_sync() {
    printf("  test_default_stream_sync... ");
    // Default stream (nullptr) should be synchronizable
    CHECK(cudaStreamSynchronize(nullptr));
    CHECK(cudaDeviceSynchronize());
    printf("PASS\n");
}

static void test_multiple_streams() {
    printf("  test_multiple_streams... ");
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
    }

    // Allocate and copy data on each stream
    const int N = 64;
    for (int i = 0; i < NUM_STREAMS; i++) {
        void* dev;
        float host_src[N], host_dst[N];
        for (int j = 0; j < N; j++) host_src[j] = (float)(i * N + j);

        CHECK(cudaMalloc(&dev, N * sizeof(float)));
        CHECK(cudaMemcpyAsync(dev, host_src, N * sizeof(float),
                              cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaStreamSynchronize(streams[i]));
        CHECK(cudaMemcpy(host_dst, dev, N * sizeof(float),
                         cudaMemcpyDeviceToHost));

        for (int j = 0; j < N; j++) {
            assert(host_dst[j] == (float)(i * N + j));
        }
        CHECK(cudaFree(dev));
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    printf("PASS\n");
}

static void test_stream_query() {
    printf("  test_stream_query... ");
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Empty stream should be complete
    cudaError_t status = cudaStreamQuery(stream);
    assert(status == cudaSuccess);

    CHECK(cudaStreamDestroy(stream));
    printf("PASS\n");
}

int main() {
    printf("=== Stream Tests ===\n");
    test_stream_create_destroy();
    test_stream_with_flags();
    test_default_stream_sync();
    test_multiple_streams();
    test_stream_query();
    printf("All stream tests PASSED\n");
    return 0;
}
