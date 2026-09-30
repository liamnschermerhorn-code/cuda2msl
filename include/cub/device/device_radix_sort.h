#pragma once

// CUB-compatible DeviceRadixSort for CUDA-to-Metal translation layer.
// Implements the two-pass CUB pattern: first call with nullptr d_temp_storage
// to get the required temp size, second call to actually sort.
// Internally uses std::sort on host-accessible unified memory.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>

namespace cub {

struct DeviceRadixSort {
    // =====================================================================
    // SortKeys -- sort an array of keys
    // =====================================================================
    template <typename KeyT>
    static cudaError_t SortKeys(
        void*          d_temp_storage,
        size_t&        temp_storage_bytes,
        const KeyT*    d_keys_in,
        KeyT*          d_keys_out,
        int            num_items,
        int            begin_bit = 0,
        int            end_bit   = sizeof(KeyT) * 8,
        cudaStream_t   stream    = nullptr)
    {
        (void)begin_bit; (void)end_bit; (void)stream;

        // Size query
        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1; // minimal non-zero
            return cudaSuccess;
        }

        if (num_items <= 0) return cudaSuccess;

        // Copy input to output, then sort output in place
        std::memcpy(d_keys_out, d_keys_in,
                    static_cast<size_t>(num_items) * sizeof(KeyT));
        std::sort(d_keys_out, d_keys_out + num_items);
        return cudaSuccess;
    }

    // SortKeysDescending
    template <typename KeyT>
    static cudaError_t SortKeysDescending(
        void*          d_temp_storage,
        size_t&        temp_storage_bytes,
        const KeyT*    d_keys_in,
        KeyT*          d_keys_out,
        int            num_items,
        int            begin_bit = 0,
        int            end_bit   = sizeof(KeyT) * 8,
        cudaStream_t   stream    = nullptr)
    {
        (void)begin_bit; (void)end_bit; (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        if (num_items <= 0) return cudaSuccess;

        std::memcpy(d_keys_out, d_keys_in,
                    static_cast<size_t>(num_items) * sizeof(KeyT));
        std::sort(d_keys_out, d_keys_out + num_items, std::greater<KeyT>());
        return cudaSuccess;
    }

    // =====================================================================
    // SortPairs -- sort key-value pairs
    // =====================================================================
    template <typename KeyT, typename ValueT>
    static cudaError_t SortPairs(
        void*           d_temp_storage,
        size_t&         temp_storage_bytes,
        const KeyT*     d_keys_in,
        KeyT*           d_keys_out,
        const ValueT*   d_values_in,
        ValueT*         d_values_out,
        int             num_items,
        int             begin_bit = 0,
        int             end_bit   = sizeof(KeyT) * 8,
        cudaStream_t    stream    = nullptr)
    {
        (void)begin_bit; (void)end_bit; (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        if (num_items <= 0) return cudaSuccess;

        size_t n = static_cast<size_t>(num_items);

        // Build index permutation sorted by key
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), size_t(0));
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return d_keys_in[a] < d_keys_in[b];
        });

        for (size_t i = 0; i < n; ++i) {
            d_keys_out[i]   = d_keys_in[idx[i]];
            d_values_out[i] = d_values_in[idx[i]];
        }
        return cudaSuccess;
    }

    // SortPairsDescending
    template <typename KeyT, typename ValueT>
    static cudaError_t SortPairsDescending(
        void*           d_temp_storage,
        size_t&         temp_storage_bytes,
        const KeyT*     d_keys_in,
        KeyT*           d_keys_out,
        const ValueT*   d_values_in,
        ValueT*         d_values_out,
        int             num_items,
        int             begin_bit = 0,
        int             end_bit   = sizeof(KeyT) * 8,
        cudaStream_t    stream    = nullptr)
    {
        (void)begin_bit; (void)end_bit; (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        if (num_items <= 0) return cudaSuccess;

        size_t n = static_cast<size_t>(num_items);

        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), size_t(0));
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return d_keys_in[a] > d_keys_in[b];
        });

        for (size_t i = 0; i < n; ++i) {
            d_keys_out[i]   = d_keys_in[idx[i]];
            d_values_out[i] = d_values_in[idx[i]];
        }
        return cudaSuccess;
    }
};

} // namespace cub
