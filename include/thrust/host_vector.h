#pragma once

// Thrust-compatible host_vector for CUDA-to-Metal translation layer.
// Thin wrapper around std::vector with Thrust-compatible interface.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace thrust {

// Forward declaration
template <typename T> class device_vector;

template <typename T>
class host_vector {
public:
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = typename std::vector<T>::iterator;
    using const_iterator  = typename std::vector<T>::const_iterator;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    host_vector() = default;
    explicit host_vector(size_type n) : vec_(n) {}
    host_vector(size_type n, const T& value) : vec_(n, value) {}

    template <typename InputIterator>
    host_vector(InputIterator first, InputIterator last) : vec_(first, last) {}

    // Construct from device_vector (copy device -> host)
    host_vector(const device_vector<T>& dv)
        : vec_(dv.begin(), dv.end()) {}

    host_vector(const host_vector&) = default;
    host_vector(host_vector&&) noexcept = default;
    host_vector& operator=(const host_vector&) = default;
    host_vector& operator=(host_vector&&) noexcept = default;

    // Assign from device_vector
    host_vector& operator=(const device_vector<T>& dv) {
        vec_.assign(dv.begin(), dv.end());
        return *this;
    }

    ~host_vector() = default;

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------

    reference       operator[](size_type i)       { return vec_[i]; }
    const_reference operator[](size_type i) const { return vec_[i]; }

    pointer       data()       { return vec_.data(); }
    const_pointer data() const { return vec_.data(); }

    reference       front()       { return vec_.front(); }
    const_reference front() const { return vec_.front(); }
    reference       back()        { return vec_.back(); }
    const_reference back()  const { return vec_.back(); }

    // -----------------------------------------------------------------------
    // Iterators
    // -----------------------------------------------------------------------

    iterator       begin()        { return vec_.begin(); }
    const_iterator begin()  const { return vec_.begin(); }
    const_iterator cbegin() const { return vec_.cbegin(); }

    iterator       end()        { return vec_.end(); }
    const_iterator end()  const { return vec_.end(); }
    const_iterator cend() const { return vec_.cend(); }

    // -----------------------------------------------------------------------
    // Capacity
    // -----------------------------------------------------------------------

    size_type size()     const { return vec_.size(); }
    bool      empty()    const { return vec_.empty(); }
    size_type capacity() const { return vec_.capacity(); }

    void reserve(size_type n) { vec_.reserve(n); }

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------

    void resize(size_type n)               { vec_.resize(n); }
    void resize(size_type n, const T& val) { vec_.resize(n, val); }

    void push_back(const T& val) { vec_.push_back(val); }
    void push_back(T&& val)      { vec_.push_back(std::move(val)); }

    void clear() { vec_.clear(); }

    void swap(host_vector& other) noexcept { vec_.swap(other.vec_); }

private:
    std::vector<T> vec_;
};

} // namespace thrust
