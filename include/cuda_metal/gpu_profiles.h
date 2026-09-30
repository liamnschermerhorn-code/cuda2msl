#pragma once

// Pre-configured NVIDIA GPU profiles for identity spoofing.
// Selected via CUDA_METAL_GPU_PROFILE environment variable.
// Set to "auto" to benchmark and pick the best match automatically.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cuda_metal {

struct GpuProfile {
    const char* name;
    const char* pci_bus_id;
    int         compute_major;
    int         compute_minor;
    size_t      vram_bytes;
    int         sm_count;
    int         cuda_cores;
    int         clock_mhz;
    int         mem_clock_mhz;
    int         mem_bus_width;
    int         l2_cache_bytes;
    int         max_threads_per_sm;
    int         driver_major;
    int         driver_minor;
    int         cuda_version_major;
    int         cuda_version_minor;
    float       tdp_watts;
    float       fp32_tflops;
};

// Profiles ordered by FP32 TFLOPS (ascending) for auto-matching

constexpr GpuProfile GPU_GTX_1650 = {
    "NVIDIA GeForce GTX 1650", "00000000:01:00.0",
    7, 5, 4ULL*1024*1024*1024, 14, 896, 1590, 6001, 128,
    1024*1024, 1024, 550, 54, 12, 4, 75.0f, 2.9f
};

constexpr GpuProfile GPU_RTX_3060 = {
    "NVIDIA GeForce RTX 3060", "00000000:01:00.0",
    8, 6, 12ULL*1024*1024*1024, 28, 3584, 1777, 7501, 192,
    3*1024*1024, 1536, 550, 54, 12, 4, 170.0f, 12.7f
};

constexpr GpuProfile GPU_RTX_3070_TI = {
    "NVIDIA GeForce RTX 3070 Ti", "00000000:01:00.0",
    8, 6, 8ULL*1024*1024*1024, 48, 6144, 1770, 9501, 256,
    4*1024*1024, 1536, 550, 54, 12, 4, 290.0f, 21.7f
};

constexpr GpuProfile GPU_RTX_3080 = {
    "NVIDIA GeForce RTX 3080", "00000000:01:00.0",
    8, 6, 10ULL*1024*1024*1024, 68, 8704, 1710, 9501, 320,
    5*1024*1024, 1536, 550, 54, 12, 4, 320.0f, 29.8f
};

constexpr GpuProfile GPU_RTX_3090 = {
    "NVIDIA GeForce RTX 3090", "00000000:01:00.0",
    8, 6, 24ULL*1024*1024*1024, 82, 10496, 1695, 9751, 384,
    6*1024*1024, 1536, 550, 54, 12, 4, 350.0f, 35.6f
};

constexpr GpuProfile GPU_RTX_4090 = {
    "NVIDIA GeForce RTX 4090", "00000000:01:00.0",
    8, 9, 24ULL*1024*1024*1024, 128, 16384, 2520, 10501, 384,
    72*1024*1024, 1536, 550, 54, 12, 4, 450.0f, 82.6f
};

constexpr GpuProfile GPU_A100 = {
    "NVIDIA A100-SXM4-80GB", "00000000:07:00.0",
    8, 0, 80ULL*1024*1024*1024, 108, 6912, 1410, 1215, 5120,
    40*1024*1024, 2048, 550, 54, 12, 4, 400.0f, 19.5f
};

constexpr GpuProfile GPU_H100 = {
    "NVIDIA H100 80GB HBM3", "00000000:07:00.0",
    9, 0, 80ULL*1024*1024*1024, 132, 16896, 1980, 1593, 5120,
    50*1024*1024, 2048, 550, 54, 12, 4, 700.0f, 66.9f
};

// All consumer profiles sorted by TFLOPS for auto-matching
// (data center cards A100/H100 excluded from auto — user must opt in)
struct ProfileEntry {
    const char* key;
    const GpuProfile* profile;
};

constexpr ProfileEntry ALL_CONSUMER_PROFILES[] = {
    {"GTX1650",   &GPU_GTX_1650},
    {"RTX3060",   &GPU_RTX_3060},
    {"RTX3070TI", &GPU_RTX_3070_TI},
    {"RTX3080",   &GPU_RTX_3080},
    {"RTX3090",   &GPU_RTX_3090},
    {"RTX4090",   &GPU_RTX_4090},
};
constexpr int NUM_CONSUMER_PROFILES = sizeof(ALL_CONSUMER_PROFILES) / sizeof(ALL_CONSUMER_PROFILES[0]);

// Pick the best-matching NVIDIA GPU for a measured TFLOPS value.
// Strategy: pick the GPU whose rated TFLOPS is closest to (but not more
// than 1.4x above) our measured performance, so we never look suspiciously
// slow OR suspiciously fast.
inline const GpuProfile* matchProfile(float measured_tflops) {
    const GpuProfile* best = &GPU_RTX_3060;  // fallback
    float best_diff = 9999.0f;

    for (int i = 0; i < NUM_CONSUMER_PROFILES; i++) {
        float rated = ALL_CONSUMER_PROFILES[i].profile->fp32_tflops;
        // We want rated to be <= 1.3x our measured (so we're at least ~77% of rated)
        // and rated should be as high as possible (maximize earning potential)
        if (rated <= measured_tflops * 1.3f) {
            float diff = measured_tflops - rated;
            if (diff < best_diff) {
                best_diff = diff;
                best = ALL_CONSUMER_PROFILES[i].profile;
            }
        }
    }
    return best;
}

inline const GpuProfile& getActiveProfile() {
    static const GpuProfile* cached = nullptr;
    if (cached) return *cached;

    const char* env = getenv("CUDA_METAL_GPU_PROFILE");
    if (!env) env = "auto";

    if (strcmp(env, "auto") == 0) {
        // Read measured TFLOPS from calibration file if available
        const char* home = getenv("HOME");
        if (home) {
            char path[512];
            snprintf(path, sizeof(path), "%s/.cuda-metal/calibration", home);
            FILE* f = fopen(path, "r");
            if (f) {
                float tflops = 0;
                if (fscanf(f, "%f", &tflops) == 1 && tflops > 0) {
                    cached = matchProfile(tflops);
                }
                fclose(f);
            }
        }
        if (!cached) cached = &GPU_RTX_3060;  // No calibration yet
    }
    else if (strcmp(env, "GTX1650") == 0)    cached = &GPU_GTX_1650;
    else if (strcmp(env, "RTX3060") == 0)    cached = &GPU_RTX_3060;
    else if (strcmp(env, "RTX3070TI") == 0)  cached = &GPU_RTX_3070_TI;
    else if (strcmp(env, "RTX3080") == 0)    cached = &GPU_RTX_3080;
    else if (strcmp(env, "RTX3090") == 0)    cached = &GPU_RTX_3090;
    else if (strcmp(env, "RTX4090") == 0)    cached = &GPU_RTX_4090;
    else if (strcmp(env, "A100") == 0)       cached = &GPU_A100;
    else if (strcmp(env, "H100") == 0)       cached = &GPU_H100;
    else                                     cached = &GPU_RTX_3060;

    return *cached;
}

} // namespace cuda_metal
