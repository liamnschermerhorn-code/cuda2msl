#pragma once

// Thrust-compatible copy primitives for CUDA-to-Metal translation layer.
// Uses std::copy on host-accessible unified memory pointers.

#include <algorithm>
#include <iterator>

#include "execution_policy.h"

namespace thrust {

// -------------------------------------------------------------------------
// thrust::copy
// -------------------------------------------------------------------------

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(InputIterator first, InputIterator last,
                    OutputIterator result) {
    return std::copy(first, last, result);
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator>
OutputIterator copy(ExecutionPolicy&&, InputIterator first,
                    InputIterator last, OutputIterator result) {
    return std::copy(first, last, result);
}

// -------------------------------------------------------------------------
// thrust::copy_n
// -------------------------------------------------------------------------

template <typename InputIterator, typename Size, typename OutputIterator>
OutputIterator copy_n(InputIterator first, Size n, OutputIterator result) {
    return std::copy_n(first, n, result);
}

template <typename ExecutionPolicy, typename InputIterator, typename Size,
          typename OutputIterator>
OutputIterator copy_n(ExecutionPolicy&&, InputIterator first, Size n,
                      OutputIterator result) {
    return std::copy_n(first, n, result);
}

// -------------------------------------------------------------------------
// thrust::copy_if
// -------------------------------------------------------------------------

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator copy_if(InputIterator first, InputIterator last,
                       OutputIterator result, Predicate pred) {
    return std::copy_if(first, last, result, pred);
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename Predicate>
OutputIterator copy_if(ExecutionPolicy&&, InputIterator first,
                       InputIterator last, OutputIterator result,
                       Predicate pred) {
    return std::copy_if(first, last, result, pred);
}

} // namespace thrust
