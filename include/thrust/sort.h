#pragma once

// Thrust-compatible sort primitives for CUDA-to-Metal translation layer.
// Uses std::sort on host-accessible unified memory pointers.

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include "execution_policy.h"

namespace thrust {

// -------------------------------------------------------------------------
// thrust::sort
// -------------------------------------------------------------------------

template <typename RandomAccessIterator>
void sort(RandomAccessIterator first, RandomAccessIterator last) {
    std::sort(first, last);
}

template <typename RandomAccessIterator, typename StrictWeakOrdering>
void sort(RandomAccessIterator first, RandomAccessIterator last,
          StrictWeakOrdering comp) {
    std::sort(first, last, comp);
}

// Overloads with execution policy (ignored -- always CPU)
template <typename ExecutionPolicy, typename RandomAccessIterator>
void sort(ExecutionPolicy&&, RandomAccessIterator first,
          RandomAccessIterator last) {
    std::sort(first, last);
}

template <typename ExecutionPolicy, typename RandomAccessIterator,
          typename StrictWeakOrdering>
void sort(ExecutionPolicy&&, RandomAccessIterator first,
          RandomAccessIterator last, StrictWeakOrdering comp) {
    std::sort(first, last, comp);
}

// -------------------------------------------------------------------------
// thrust::stable_sort
// -------------------------------------------------------------------------

template <typename RandomAccessIterator>
void stable_sort(RandomAccessIterator first, RandomAccessIterator last) {
    std::stable_sort(first, last);
}

template <typename RandomAccessIterator, typename StrictWeakOrdering>
void stable_sort(RandomAccessIterator first, RandomAccessIterator last,
                 StrictWeakOrdering comp) {
    std::stable_sort(first, last, comp);
}

template <typename ExecutionPolicy, typename RandomAccessIterator>
void stable_sort(ExecutionPolicy&&, RandomAccessIterator first,
                 RandomAccessIterator last) {
    std::stable_sort(first, last);
}

template <typename ExecutionPolicy, typename RandomAccessIterator,
          typename StrictWeakOrdering>
void stable_sort(ExecutionPolicy&&, RandomAccessIterator first,
                 RandomAccessIterator last, StrictWeakOrdering comp) {
    std::stable_sort(first, last, comp);
}

// -------------------------------------------------------------------------
// thrust::sort_by_key
// -------------------------------------------------------------------------

template <typename KeyIterator, typename ValueIterator>
void sort_by_key(KeyIterator keys_first, KeyIterator keys_last,
                 ValueIterator values_first) {
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    auto n = std::distance(keys_first, keys_last);
    if (n <= 0) return;

    // Build index permutation, sort indices by key, then reorder both
    std::vector<std::size_t> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return *(keys_first + a) < *(keys_first + b);
    });

    // Apply permutation in-place via temporary buffers
    using val_type = typename std::iterator_traits<ValueIterator>::value_type;
    std::vector<key_type> sorted_keys(static_cast<std::size_t>(n));
    std::vector<val_type> sorted_vals(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        sorted_keys[i] = *(keys_first   + idx[i]);
        sorted_vals[i] = *(values_first  + idx[i]);
    }
    std::copy(sorted_keys.begin(), sorted_keys.end(), keys_first);
    std::copy(sorted_vals.begin(), sorted_vals.end(), values_first);
}

template <typename KeyIterator, typename ValueIterator,
          typename StrictWeakOrdering>
void sort_by_key(KeyIterator keys_first, KeyIterator keys_last,
                 ValueIterator values_first, StrictWeakOrdering comp) {
    auto n = std::distance(keys_first, keys_last);
    if (n <= 0) return;

    std::vector<std::size_t> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return comp(*(keys_first + a), *(keys_first + b));
    });

    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    using val_type = typename std::iterator_traits<ValueIterator>::value_type;
    std::vector<key_type> sorted_keys(static_cast<std::size_t>(n));
    std::vector<val_type> sorted_vals(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        sorted_keys[i] = *(keys_first   + idx[i]);
        sorted_vals[i] = *(values_first  + idx[i]);
    }
    std::copy(sorted_keys.begin(), sorted_keys.end(), keys_first);
    std::copy(sorted_vals.begin(), sorted_vals.end(), values_first);
}

// Overloads with execution policy
template <typename ExecutionPolicy, typename KeyIterator,
          typename ValueIterator>
void sort_by_key(ExecutionPolicy&&, KeyIterator keys_first,
                 KeyIterator keys_last, ValueIterator values_first) {
    sort_by_key(keys_first, keys_last, values_first);
}

template <typename ExecutionPolicy, typename KeyIterator,
          typename ValueIterator, typename StrictWeakOrdering>
void sort_by_key(ExecutionPolicy&&, KeyIterator keys_first,
                 KeyIterator keys_last, ValueIterator values_first,
                 StrictWeakOrdering comp) {
    sort_by_key(keys_first, keys_last, values_first, comp);
}

} // namespace thrust
