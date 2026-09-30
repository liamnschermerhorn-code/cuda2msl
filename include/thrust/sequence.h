#pragma once

// Thrust-compatible sequence primitive for CUDA-to-Metal translation layer.
// Uses std::iota on host-accessible unified memory pointers.

#include <iterator>
#include <numeric>

#include "execution_policy.h"

namespace thrust {

// -------------------------------------------------------------------------
// thrust::sequence  (fills [first, last) with init, init+step, ...)
// -------------------------------------------------------------------------

// sequence(first, last)  -- fills with 0, 1, 2, ...
template <typename ForwardIterator>
void sequence(ForwardIterator first, ForwardIterator last) {
    using T = typename std::iterator_traits<ForwardIterator>::value_type;
    std::iota(first, last, T{});
}

// sequence(first, last, init)  -- fills with init, init+1, ...
template <typename ForwardIterator, typename T>
void sequence(ForwardIterator first, ForwardIterator last, T init) {
    std::iota(first, last, init);
}

// sequence(first, last, init, step)  -- fills with init, init+step, ...
template <typename ForwardIterator, typename T>
void sequence(ForwardIterator first, ForwardIterator last, T init, T step) {
    for (; first != last; ++first) {
        *first = init;
        init += step;
    }
}

// Overloads with execution policy
template <typename ExecutionPolicy, typename ForwardIterator>
void sequence(ExecutionPolicy&&, ForwardIterator first,
              ForwardIterator last) {
    thrust::sequence(first, last);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
void sequence(ExecutionPolicy&&, ForwardIterator first,
              ForwardIterator last, T init) {
    thrust::sequence(first, last, init);
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
void sequence(ExecutionPolicy&&, ForwardIterator first,
              ForwardIterator last, T init, T step) {
    thrust::sequence(first, last, init, step);
}

} // namespace thrust
