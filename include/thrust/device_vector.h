#pragma once

// Thrust-compatible device_vector for CUDA-to-Metal translation layer.
// Wraps cudaMalloc/cudaFree. Because Metal uses unified memory, the raw
// device pointer is directly host-accessible -- all STL algorithms work on it.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace thrust {

template <typename T>
class device_vector {
public:
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = const T*;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    // -----------------------------------------------------------------------
    // Construction / Destruction
    // -----------------------------------------------------------------------

    device_vector() : data_(nullptr), size_(0), capacity_(0) {}

    explicit device_vector(size_type n) : data_(nullptr), size_(0), capacity_(0) {
        allocate_and_init(n);
    }

    device_vector(size_type n, const T& value) : data_(nullptr), size_(0), capacity_(0) {
        allocate_and_init(n);
        std::fill(data_, data_ + size_, value);
    }

    // Construct from iterator range
    template <typename InputIterator>
    device_vector(InputIterator first, InputIterator last)
        : data_(nullptr), size_(0), capacity_(0) {
        size_type n = static_cast<size_type>(std::distance(first, last));
        allocate_and_init(n);
        std::copy(first, last, data_);
    }

    // Copy constructor
    device_vector(const device_vector& other) : data_(nullptr), size_(0), capacity_(0) {
        allocate_and_init(other.size_);
        if (size_ > 0) {
            std::memcpy(data_, other.data_, size_ * sizeof(T));
        }
    }

    // Move constructor
    device_vector(device_vector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_     = nullptr;
        other.size_     = 0;
        other.capacity_ = 0;
    }

    // Copy assignment
    device_vector& operator=(const device_vector& other) {
        if (this != &other) {
            free_storage();
            allocate_and_init(other.size_);
            if (size_ > 0) {
                std::memcpy(data_, other.data_, size_ * sizeof(T));
            }
        }
        return *this;
    }

    // Move assignment
    device_vector& operator=(device_vector&& other) noexcept {
        if (this != &other) {
            free_storage();
            data_           = other.data_;
            size_           = other.size_;
            capacity_       = other.capacity_;
            other.data_     = nullptr;
            other.size_     = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    ~device_vector() { free_storage(); }

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------

    reference       operator[](size_type i)       { return data_[i]; }
    const_reference operator[](size_type i) const { return data_[i]; }

    pointer       data()       { return data_; }
    const_pointer data() const { return data_; }

    // -----------------------------------------------------------------------
    // Iterators
    // -----------------------------------------------------------------------

    iterator       begin()        { return data_; }
    const_iterator begin()  const { return data_; }
    const_iterator cbegin() const { return data_; }

    iterator       end()        { return data_ + size_; }
    const_iterator end()  const { return data_ + size_; }
    const_iterator cend() const { return data_ + size_; }

    // -----------------------------------------------------------------------
    // Capacity
    // -----------------------------------------------------------------------

    size_type size()     const { return size_; }
    bool      empty()    const { return size_ == 0; }
    size_type capacity() const { return capacity_; }

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------

    void resize(size_type new_size) {
        if (new_size <= capacity_) {
            // Zero-initialize new elements if growing within capacity
            if (new_size > size_) {
                std::memset(data_ + size_, 0, (new_size - size_) * sizeof(T));
            }
            size_ = new_size;
            return;
        }
        // Need reallocation
        T* new_data = nullptr;
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&new_data),
                                     new_size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("thrust::device_vector::resize: cudaMalloc failed");
        }
        if (data_ && size_ > 0) {
            std::memcpy(new_data, data_, size_ * sizeof(T));
        }
        if (new_size > size_) {
            std::memset(new_data + size_, 0, (new_size - size_) * sizeof(T));
        }
        free_storage();
        data_     = new_data;
        size_     = new_size;
        capacity_ = new_size;
    }

    void resize(size_type new_size, const T& value) {
        size_type old_size = size_;
        resize(new_size);
        if (new_size > old_size) {
            std::fill(data_ + old_size, data_ + new_size, value);
        }
    }

    void clear() { size_ = 0; }

    void swap(device_vector& other) noexcept {
        std::swap(data_,     other.data_);
        std::swap(size_,     other.size_);
        std::swap(capacity_, other.capacity_);
    }

    // -----------------------------------------------------------------------
    // Raw-pointer access (CUDA API compatibility)
    // -----------------------------------------------------------------------

    // thrust::raw_pointer_cast equivalent
    pointer raw_pointer() { return data_; }

private:
    T*        data_;
    size_type size_;
    size_type capacity_;

    void allocate_and_init(size_type n) {
        if (n == 0) {
            data_     = nullptr;
            size_     = 0;
            capacity_ = 0;
            return;
        }
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_),
                                     n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("thrust::device_vector: cudaMalloc failed");
        }
        size_     = n;
        capacity_ = n;
        std::memset(data_, 0, n * sizeof(T));
    }

    void free_storage() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
        }
        size_     = 0;
        capacity_ = 0;
    }
};

// Free-standing raw_pointer_cast (matches real Thrust API)
template <typename T>
T* raw_pointer_cast(T* ptr) { return ptr; }

template <typename T>
T* raw_pointer_cast(device_vector<T>& v) { return v.data(); }

} // namespace thrust
