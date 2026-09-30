#pragma once

// Thrust-compatible fill primitives for CUDA-to-Metal translation layer.
// Uses std::fill / std::fill_n on host-accessible unified memory pointers.

#include <algorithm>
#include <iterator>

#include "execution_policy.h"

namespace thrust {

// -------------------------------------------------------------------------
// thrust::fill
// -------------------------------------------------------------------------

template <typename ForwardIterator, typename T>
void fill(ForwardIterator first, ForwardIterator last, const T& value) {
    std::fill(first, last, value);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
void fill(ExecutionPolicy&&, ForwardIterator first, ForwardIterator last,
          const T& value) {
    std::fill(first, last, value);
}

// -------------------------------------------------------------------------
// thrust::fill_n
// -------------------------------------------------------------------------

template <typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(OutputIterator first, Size n, const T& value) {
    return std::fill_n(first, n, value);
}

template <typename ExecutionPolicy, typename OutputIterator, typename Size,
          typename T>
OutputIterator fill_n(ExecutionPolicy&&, OutputIterator first, Size n,
                      const T& value) {
    return std::fill_n(first, n, value);
}

} // namespace thrust
