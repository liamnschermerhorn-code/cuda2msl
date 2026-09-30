#pragma once

// Thrust-compatible reduce primitives for CUDA-to-Metal translation layer.
// Uses std::accumulate on host-accessible unified memory pointers.

#include <functional>
#include <iterator>
#include <numeric>

#include "execution_policy.h"
#include "functional.h"

namespace thrust {

// -------------------------------------------------------------------------
// thrust::reduce
// -------------------------------------------------------------------------

// reduce(first, last) -> sum with value-initialized identity
template <typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type
reduce(InputIterator first, InputIterator last) {
    using T = typename std::iterator_traits<InputIterator>::value_type;
    return std::accumulate(first, last, T{});
}

// reduce(first, last, init)
template <typename InputIterator, typename T>
T reduce(InputIterator first, InputIterator last, T init) {
    return std::accumulate(first, last, init);
}

// reduce(first, last, init, binary_op)
template <typename InputIterator, typename T, typename BinaryFunction>
T reduce(InputIterator first, InputIterator last, T init,
         BinaryFunction binary_op) {
    return std::accumulate(first, last, init, binary_op);
}

// Overloads with execution policy
template <typename ExecutionPolicy, typename InputIterator>
typename std::iterator_traits<InputIterator>::value_type
reduce(ExecutionPolicy&&, InputIterator first, InputIterator last) {
    return reduce(first, last);
}

template <typename ExecutionPolicy, typename InputIterator, typename T>
T reduce(ExecutionPolicy&&, InputIterator first, InputIterator last, T init) {
    return std::accumulate(first, last, init);
}

template <typename ExecutionPolicy, typename InputIterator, typename T,
          typename BinaryFunction>
T reduce(ExecutionPolicy&&, InputIterator first, InputIterator last, T init,
         BinaryFunction binary_op) {
    return std::accumulate(first, last, init, binary_op);
}

} // namespace thrust
