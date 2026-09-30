#pragma once

// Thrust-compatible transform primitives for CUDA-to-Metal translation layer.
// Uses std::transform on host-accessible unified memory pointers.

#include <algorithm>
#include <iterator>

#include "execution_policy.h"

namespace thrust {

// -------------------------------------------------------------------------
// Unary transform: result[i] = op(first[i])
// -------------------------------------------------------------------------

template <typename InputIterator, typename OutputIterator,
          typename UnaryFunction>
OutputIterator transform(InputIterator first, InputIterator last,
                         OutputIterator result, UnaryFunction op) {
    return std::transform(first, last, result, op);
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename UnaryFunction>
OutputIterator transform(ExecutionPolicy&&, InputIterator first,
                         InputIterator last, OutputIterator result,
                         UnaryFunction op) {
    return std::transform(first, last, result, op);
}

// -------------------------------------------------------------------------
// Binary transform: result[i] = op(first1[i], first2[i])
// -------------------------------------------------------------------------

template <typename InputIterator1, typename InputIterator2,
          typename OutputIterator, typename BinaryFunction>
OutputIterator transform(InputIterator1 first1, InputIterator1 last1,
                         InputIterator2 first2, OutputIterator result,
                         BinaryFunction op) {
    return std::transform(first1, last1, first2, result, op);
}

template <typename ExecutionPolicy, typename InputIterator1,
          typename InputIterator2, typename OutputIterator,
          typename BinaryFunction>
OutputIterator transform(ExecutionPolicy&&, InputIterator1 first1,
                         InputIterator1 last1, InputIterator2 first2,
                         OutputIterator result, BinaryFunction op) {
    return std::transform(first1, last1, first2, result, op);
}

} // namespace thrust
