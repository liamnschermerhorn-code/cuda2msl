#include "cuda_metal/secure_weight_pool.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>

#include <unistd.h>

using namespace cuda_metal;

// ── Helpers ────────────────────────────────────────────────────────────────

static void fill_pattern(void* ptr, size_t bytes) {
    auto* dst = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < bytes; ++i) {
        dst[i] = static_cast<uint8_t>(i & 0xFF);
    }
}

static bool verify_pattern(const void* ptr, size_t bytes) {
    auto* src = static_cast<const uint8_t*>(ptr);
    for (size_t i = 0; i < bytes; ++i) {
        if (src[i] != static_cast<uint8_t>(i & 0xFF)) return false;
    }
    return true;
}

// ── Test: basic allocate / access / release ────────────────────────────────

static void test_basic_lifecycle() {
    printf("  test_basic_lifecycle ... ");

    auto& pool = SecureWeightPool::instance();
    size_t weight_size = 1024 * 1024;  // 1 MiB

    auto handle = pool.allocate(weight_size);
    assert(handle.id != 0);
    assert(handle.host_ptr != nullptr);
    assert(handle.weight_bytes == weight_size);

    // Write a pattern through host_ptr (zero-copy).
    fill_pattern(handle.host_ptr, weight_size);
    assert(verify_pattern(handle.host_ptr, weight_size));

    // Access via get_host_ptr should return the same pointer.
    void* ptr = pool.get_host_ptr(handle.id);
    assert(ptr == handle.host_ptr);

    // Metal buffer should be non-null.
    void* mtl_buf = pool.get_metal_buffer(handle.id);
    assert(mtl_buf != nullptr);

    // Stats.
    assert(pool.active_count() == 1);
    assert(pool.total_bytes_managed() == weight_size);

    pool.release(handle.id);
    assert(pool.active_count() == 0);

    printf("PASSED\n");
}

// ── Test: multiple buffers ─────────────────────────────────────────────────

static void test_multiple_buffers() {
    printf("  test_multiple_buffers ... ");

    auto& pool = SecureWeightPool::instance();
    constexpr int N = 8;
    SecureWeightHandle handles[N];
    size_t sizes[N] = {4096, 65536, 1048576, 256, 512, 2097152, 128, 8192};

    for (int i = 0; i < N; ++i) {
        handles[i] = pool.allocate(sizes[i]);
        assert(handles[i].id != 0);
        fill_pattern(handles[i].host_ptr, sizes[i]);
    }

    assert(pool.active_count() == N);

    // Verify each buffer's data independently.
    for (int i = 0; i < N; ++i) {
        assert(verify_pattern(handles[i].host_ptr, sizes[i]));
    }

    // Release in reverse order.
    for (int i = N - 1; i >= 0; --i) {
        pool.release(handles[i].id);
    }

    assert(pool.active_count() == 0);

    printf("PASSED\n");
}

// ── Test: integrity check passes on clean buffers ──────────────────────────

static void test_integrity_clean() {
    printf("  test_integrity_clean ... ");

    auto& pool = SecureWeightPool::instance();
    auto h1 = pool.allocate(4096);
    auto h2 = pool.allocate(8192);

    fill_pattern(h1.host_ptr, 4096);
    fill_pattern(h2.host_ptr, 8192);

    assert(pool.check_integrity() == true);

    pool.release_all();

    printf("PASSED\n");
}

// ── Test: integrity detects canary corruption ──────────────────────────────

static void test_integrity_canary_violation() {
    printf("  test_integrity_canary_violation ... ");

    auto& pool = SecureWeightPool::instance();
    auto handle = pool.allocate(4096);

    // Corrupt the leading guard by writing before host_ptr.
    auto* guard = static_cast<uint8_t*>(handle.host_ptr) - 8;
    *guard = 0xDE;

    std::atomic<bool> violation_seen{false};
    pool.start_heartbeat(50, [&](uint64_t id, const char* reason) {
        (void)reason;
        if (id == handle.id) {
            violation_seen.store(true);
        }
    });

    // Wait for the watchdog to catch it.
    for (int i = 0; i < 20 && !violation_seen.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    pool.stop_heartbeat();

    // After violation, the pool should have wiped everything.
    assert(violation_seen.load());
    assert(pool.active_count() == 0);

    printf("PASSED\n");
}

// ── Test: file loading ─────────────────────────────────────────────────────

static void test_load_from_file() {
    printf("  test_load_from_file ... ");

    // Write a temp file with known contents.
    std::string tmp_path = "/tmp/test_secure_weights.bin";
    {
        std::ofstream ofs(tmp_path, std::ios::binary);
        assert(ofs.good());
        constexpr size_t n = 8192;
        for (size_t i = 0; i < n; ++i) {
            uint8_t byte = static_cast<uint8_t>(i & 0xFF);
            ofs.write(reinterpret_cast<const char*>(&byte), 1);
        }
    }

    auto& pool = SecureWeightPool::instance();
    auto handle = pool.load_from_file(tmp_path.c_str(), 0, 8192);
    assert(handle.id != 0);
    assert(handle.weight_bytes == 8192);
    assert(verify_pattern(handle.host_ptr, 8192));

    // Load with offset.
    auto handle2 = pool.load_from_file(tmp_path.c_str(), 1024, 4096);
    assert(handle2.id != 0);
    assert(handle2.weight_bytes == 4096);
    // The data at offset 1024 should match our pattern starting at byte 0 of that offset.
    auto* data = static_cast<const uint8_t*>(handle2.host_ptr);
    for (size_t i = 0; i < 4096; ++i) {
        assert(data[i] == static_cast<uint8_t>((i + 1024) & 0xFF));
    }

    pool.release_all();
    unlink(tmp_path.c_str());

    printf("PASSED\n");
}

// ── Test: DeepSeek-V3 scale allocation ─────────────────────────────────────

static void test_deepseek_v3_scale() {
    printf("  test_deepseek_v3_scale ... ");

    auto& pool = SecureWeightPool::instance();

    // Simulate loading a MoE expert shard: 256 MiB.
    constexpr size_t expert_shard = 256ULL * 1024 * 1024;
    auto handle = pool.allocate(expert_shard);
    assert(handle.id != 0);
    assert(handle.weight_bytes == expert_shard);

    // Verify integrity at scale.
    assert(pool.check_integrity() == true);

    // Zero-copy: write directly.
    memset(handle.host_ptr, 0x42, expert_shard);
    auto* check = static_cast<uint8_t*>(handle.host_ptr);
    assert(check[0] == 0x42);
    assert(check[expert_shard - 1] == 0x42);

    pool.release(handle.id);

    printf("PASSED\n");
}

// ── main ───────────────────────────────────────────────────────────────────

int main() {
    printf("=== SecureWeightPool Tests ===\n");

    test_basic_lifecycle();
    test_multiple_buffers();
    test_integrity_clean();
    test_integrity_canary_violation();
    test_load_from_file();
    test_deepseek_v3_scale();

    printf("=== All SecureWeightPool tests PASSED ===\n");
    return 0;
}
