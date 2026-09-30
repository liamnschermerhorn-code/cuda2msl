#pragma once

// CUB-compatible DeviceScan for CUDA-to-Metal translation layer.
// Implements the two-pass CUB pattern: first call with nullptr d_temp_storage
// returns required temp size; second call performs the scan.
// Internally uses direct CPU iteration on host-accessible unified memory.

#include <cuda_runtime.h>

#include <cstddef>
#include <iterator>

namespace cub {

struct DeviceScan {
    // =====================================================================
    // InclusiveSum
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t InclusiveSum(
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
        if (num_items <= 0) return cudaSuccess;

        T running = d_in[0];
        d_out[0] = running;
        for (int i = 1; i < num_items; ++i) {
            running = running + d_in[i];
            d_out[i] = running;
        }
        return cudaSuccess;
    }

    // =====================================================================
    // ExclusiveSum
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t ExclusiveSum(
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
        if (num_items <= 0) return cudaSuccess;

        T running = T{};
        for (int i = 0; i < num_items; ++i) {
            T val = d_in[i];
            d_out[i] = running;
            running = running + val;
        }
        return cudaSuccess;
    }

    // =====================================================================
    // InclusiveScan (generic binary op)
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT,
              typename ScanOpT>
    static cudaError_t InclusiveScan(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        ScanOpT          scan_op,
        int              num_items,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        if (num_items <= 0) return cudaSuccess;

        T running = d_in[0];
        d_out[0] = running;
        for (int i = 1; i < num_items; ++i) {
            running = scan_op(running, d_in[i]);
            d_out[i] = running;
        }
        return cudaSuccess;
    }

    // =====================================================================
    // ExclusiveScan (generic binary op with initial value)
    // =====================================================================
    template <typename InputIteratorT, typename OutputIteratorT,
              typename ScanOpT, typename InitValueT>
    static cudaError_t ExclusiveScan(
        void*            d_temp_storage,
        size_t&          temp_storage_bytes,
        InputIteratorT   d_in,
        OutputIteratorT  d_out,
        ScanOpT          scan_op,
        InitValueT       init_value,
        int              num_items,
        cudaStream_t     stream = nullptr)
    {
        (void)stream;

        if (d_temp_storage == nullptr) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }

        if (num_items <= 0) return cudaSuccess;

        InitValueT running = init_value;
        for (int i = 0; i < num_items; ++i) {
            auto val = d_in[i];
            d_out[i] = running;
            running = scan_op(running, val);
        }
        return cudaSuccess;
    }
};

} // namespace cub
