#pragma once

// Thrust-compatible counting_iterator for CUDA-to-Metal translation layer.
// A random-access iterator that returns incrementing values starting from
// an initial value, without storing them in memory.

#include <cstddef>
#include <iterator>

namespace thrust {

template <typename T>
class counting_iterator {
public:
    // Iterator traits
    using value_type        = T;
    using reference         = T;          // returns by value, not actual ref
    using pointer           = const T*;
    using difference_type   = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    // Construction
    counting_iterator() : value_(T{}) {}
    explicit counting_iterator(T value) : value_(value) {}

    // Dereference
    reference operator*()  const { return value_; }
    reference operator[](difference_type n) const { return value_ + static_cast<T>(n); }

    // Increment / Decrement
    counting_iterator& operator++()    { ++value_; return *this; }
    counting_iterator  operator++(int) { auto tmp = *this; ++value_; return tmp; }
    counting_iterator& operator--()    { --value_; return *this; }
    counting_iterator  operator--(int) { auto tmp = *this; --value_; return tmp; }

    // Arithmetic
    counting_iterator& operator+=(difference_type n) { value_ += static_cast<T>(n); return *this; }
    counting_iterator& operator-=(difference_type n) { value_ -= static_cast<T>(n); return *this; }

    counting_iterator operator+(difference_type n) const {
        return counting_iterator(value_ + static_cast<T>(n));
    }
    counting_iterator operator-(difference_type n) const {
        return counting_iterator(value_ - static_cast<T>(n));
    }
    difference_type operator-(const counting_iterator& other) const {
        return static_cast<difference_type>(value_) - static_cast<difference_type>(other.value_);
    }

    // Comparisons
    bool operator==(const counting_iterator& other) const { return value_ == other.value_; }
    bool operator!=(const counting_iterator& other) const { return value_ != other.value_; }
    bool operator< (const counting_iterator& other) const { return value_ <  other.value_; }
    bool operator<=(const counting_iterator& other) const { return value_ <= other.value_; }
    bool operator> (const counting_iterator& other) const { return value_ >  other.value_; }
    bool operator>=(const counting_iterator& other) const { return value_ >= other.value_; }

private:
    T value_;
};

// n + iter
template <typename T>
counting_iterator<T> operator+(typename counting_iterator<T>::difference_type n,
                               const counting_iterator<T>& it) {
    return it + n;
}

// Factory function
template <typename T>
counting_iterator<T> make_counting_iterator(T value) {
    return counting_iterator<T>(value);
}

} // namespace thrust
