#pragma once

// Thrust-compatible scan (prefix sum) primitives for CUDA-to-Metal translation
// layer. Uses C++17 <numeric> inclusive_scan / exclusive_scan when available,
// otherwise falls back to simple loops.

#include <functional>
#include <iterator>
#include <numeric>

#include "execution_policy.h"
#include "functional.h"

namespace thrust {

// -------------------------------------------------------------------------
// thrust::inclusive_scan
// -------------------------------------------------------------------------

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                             OutputIterator result) {
    using T = typename std::iterator_traits<InputIterator>::value_type;
    if (first == last) return result;
    T acc = *first;
    *result = acc;
    ++first; ++result;
    for (; first != last; ++first, ++result) {
        acc = acc + *first;
        *result = acc;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator,
          typename BinaryFunction>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                             OutputIterator result, BinaryFunction binary_op) {
    using T = typename std::iterator_traits<InputIterator>::value_type;
    if (first == last) return result;
    T acc = *first;
    *result = acc;
    ++first; ++result;
    for (; first != last; ++first, ++result) {
        acc = binary_op(acc, *first);
        *result = acc;
    }
    return result;
}

// With execution policy
template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator>
OutputIterator inclusive_scan(ExecutionPolicy&&, InputIterator first,
                             InputIterator last, OutputIterator result) {
    return thrust::inclusive_scan(first, last, result);
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename BinaryFunction>
OutputIterator inclusive_scan(ExecutionPolicy&&, InputIterator first,
                             InputIterator last, OutputIterator result,
                             BinaryFunction binary_op) {
    return thrust::inclusive_scan(first, last, result, binary_op);
}

// -------------------------------------------------------------------------
// thrust::exclusive_scan
// -------------------------------------------------------------------------

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result) {
    using T = typename std::iterator_traits<InputIterator>::value_type;
    return thrust::exclusive_scan(first, last, result, T{});
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, T init) {
    if (first == last) return result;
    T acc = init;
    for (; first != last; ++first, ++result) {
        T val = *first;
        *result = acc;
        acc = acc + val;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator, typename T,
          typename BinaryFunction>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, T init,
                              BinaryFunction binary_op) {
    if (first == last) return result;
    T acc = init;
    for (; first != last; ++first, ++result) {
        T val = *first;
        *result = acc;
        acc = binary_op(acc, val);
    }
    return result;
}

// With execution policy
template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator>
OutputIterator exclusive_scan(ExecutionPolicy&&, InputIterator first,
                              InputIterator last, OutputIterator result) {
    return thrust::exclusive_scan(first, last, result);
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename T>
OutputIterator exclusive_scan(ExecutionPolicy&&, InputIterator first,
                              InputIterator last, OutputIterator result,
                              T init) {
    return thrust::exclusive_scan(first, last, result, init);
}

template <typename ExecutionPolicy, typename InputIterator,
          typename OutputIterator, typename T, typename BinaryFunction>
OutputIterator exclusive_scan(ExecutionPolicy&&, InputIterator first,
                              InputIterator last, OutputIterator result,
                              T init, BinaryFunction binary_op) {
    return thrust::exclusive_scan(first, last, result, init, binary_op);
}

} // namespace thrust
