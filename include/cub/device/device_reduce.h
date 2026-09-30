#pragma once

// CUB-compatible DeviceReduce for CUDA-to-Metal translation layer.
// Implements the two-pass CUB pattern: first call with nullptr d_temp_storage
// returns required temp size; second call performs the reduction.
// Internally uses direct CPU iteration on host-accessible unified memory.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

namespace cub {

struct DeviceReduce {
    // =====================================================================
    // Sum
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t Sum(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        int              num_items,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        T sum = T{};
        for (int i = 0; i < num_items; ++i) {
            sum = sum + d_in[i];
        }
        *d_out = sum;
        return cudaSuccess;
    }

    // =====================================================================
    // Reduce (generic binary op)
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT,
              typename ReductionOpT, typename T>
    static cudaError_t Reduce(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        int              num_items,
        ReductionOpT     reduction_op,
        T                init,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        T result = init;
        for (int i = 0; i < num_items; ++i) {
            result = reduction_op(result, d_in[i]);
        }
        *d_out = result;
        return cudaSuccess;
    }

    // =====================================================================
    // Min
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t Min(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        int              num_items,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        if (num_items <= 0) {
            *d_out = std::numeric_limits<T>::max();
            return cudaSuccess;
        }

        T min_val = d_in[0];
        for (int i = 1; i < num_items; ++i) {
            if (d_in[i] < min_val) min_val = d_in[i];
        }
        *d_out = min_val;
        return cudaSuccess;
    }

    // =====================================================================
    // Max
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t Max(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        int              num_items,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        if (num_items <= 0) {
            *d_out = std::numeric_limits<T>::lowest();
            return cudaSuccess;
        }

        T max_val = d_in[0];
        for (int i = 1; i < num_items; ++i) {
            if (d_in[i] > max_val) max_val = d_in[i];
        }
        *d_out = max_val;
        return cudaSuccess;
    }

    // =====================================================================
    // ArgMin -- outputs KeyValuePair {index, value}
    // =====================================================================

    // CUB-compatible key-value pair
    template <typename KeyT, typename ValueT>
    struct KeyValuePair {
        KeyT   key;
        ValueT value;
    };

    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t ArgMin(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        int              num_items,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using KVP = KeyValuePair<int, T>;

        if (num_items <= 0) {
            *d_out = KVP{1, std::numeric_limits<T>::max()};
            return cudaSuccess;
        }

        int best_idx = 0;
        T   best_val = d_in[0];
        for (int i = 1; i < num_items; ++i) {
            if (d_in[i] < best_val) {
                best_val = d_in[i];
                best_idx = i;
            }
        }
        *d_out = KVP{best_idx, best_val};
        return cudaSuccess;
    }

    // =====================================================================
    // ArgMax
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t ArgMax(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        int              num_items,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using KVP = KeyValuePair<int, T>;

        if (num_items <= 0) {
            *d_out = KVP{1, std::numeric_limits<T>::lowest()};
            return cudaSuccess;
        }

        int best_idx = 0;
        T   best_val = d_in[0];
        for (int i = 1; i < num_items; ++i) {
            if (d_in[i] > best_val) {
                best_val = d_in[i];
                best_idx = i;
            }
        }
        *d_out = KVP{best_idx, best_val};
        return cudaSuccess;
    }
};

} // namespace cub
