#pragma once

// Thrust-compatible function objects for CUDA-to-Metal translation layer.

namespace thrust {

template <typename T = void>
struct plus {
    constexpr T operator()(const T& a, const T& b) const { return a + b; }
};

template <>
struct plus<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a + b) {
        return a + b;
    }
};

template <typename T = void>
struct minus {
    constexpr T operator()(const T& a, const T& b) const { return a - b; }
};

template <>
struct minus<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a - b) {
        return a - b;
    }
};

template <typename T = void>
struct multiplies {
    constexpr T operator()(const T& a, const T& b) const { return a * b; }
};

template <>
struct multiplies<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a * b) {
        return a * b;
    }
};

template <typename T = void>
struct divides {
    constexpr T operator()(const T& a, const T& b) const { return a / b; }
};

template <>
struct divides<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a / b) {
        return a / b;
    }
};

template <typename T = void>
struct modulus {
    constexpr T operator()(const T& a, const T& b) const { return a % b; }
};

template <>
struct modulus<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a % b) {
        return a % b;
    }
};

template <typename T = void>
struct negate {
    constexpr T operator()(const T& a) const { return -a; }
};

template <>
struct negate<void> {
    template <typename T>
    constexpr auto operator()(T&& a) const -> decltype(-a) {
        return -a;
    }
};

template <typename T = void>
struct equal_to {
    constexpr bool operator()(const T& a, const T& b) const { return a == b; }
};

template <>
struct equal_to<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a == b) {
        return a == b;
    }
};

template <typename T = void>
struct not_equal_to {
    constexpr bool operator()(const T& a, const T& b) const { return a != b; }
};

template <>
struct not_equal_to<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a != b) {
        return a != b;
    }
};

template <typename T = void>
struct greater {
    constexpr bool operator()(const T& a, const T& b) const { return a > b; }
};

template <>
struct greater<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a > b) {
        return a > b;
    }
};

template <typename T = void>
struct less {
    constexpr bool operator()(const T& a, const T& b) const { return a < b; }
};

template <>
struct less<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a < b) {
        return a < b;
    }
};

template <typename T = void>
struct greater_equal {
    constexpr bool operator()(const T& a, const T& b) const { return a >= b; }
};

template <>
struct greater_equal<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a >= b) {
        return a >= b;
    }
};

template <typename T = void>
struct less_equal {
    constexpr bool operator()(const T& a, const T& b) const { return a <= b; }
};

template <>
struct less_equal<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a <= b) {
        return a <= b;
    }
};

template <typename T = void>
struct logical_and {
    constexpr bool operator()(const T& a, const T& b) const { return a && b; }
};

template <>
struct logical_and<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a && b) {
        return a && b;
    }
};

template <typename T = void>
struct logical_or {
    constexpr bool operator()(const T& a, const T& b) const { return a || b; }
};

template <>
struct logical_or<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a || b) {
        return a || b;
    }
};

template <typename T = void>
struct logical_not {
    constexpr bool operator()(const T& a) const { return !a; }
};

template <>
struct logical_not<void> {
    template <typename T>
    constexpr auto operator()(T&& a) const -> decltype(!a) {
        return !a;
    }
};

template <typename T = void>
struct bit_and {
    constexpr T operator()(const T& a, const T& b) const { return a & b; }
};

template <>
struct bit_and<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a & b) {
        return a & b;
    }
};

template <typename T = void>
struct bit_or {
    constexpr T operator()(const T& a, const T& b) const { return a | b; }
};

template <>
struct bit_or<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a | b) {
        return a | b;
    }
};

template <typename T = void>
struct bit_xor {
    constexpr T operator()(const T& a, const T& b) const { return a ^ b; }
};

template <>
struct bit_xor<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const -> decltype(a ^ b) {
        return a ^ b;
    }
};

// identity functor (commonly used as a projection)
template <typename T = void>
struct identity {
    constexpr const T& operator()(const T& a) const { return a; }
};

template <>
struct identity<void> {
    template <typename T>
    constexpr T&& operator()(T&& a) const { return static_cast<T&&>(a); }
};

// maximum / minimum (Thrust-specific, not in std::)
template <typename T = void>
struct maximum {
    constexpr const T& operator()(const T& a, const T& b) const {
        return a > b ? a : b;
    }
};

template <>
struct maximum<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(a > b ? a : b) {
        return a > b ? a : b;
    }
};

template <typename T = void>
struct minimum {
    constexpr const T& operator()(const T& a, const T& b) const {
        return a < b ? a : b;
    }
};

template <>
struct minimum<void> {
    template <typename T, typename U>
    constexpr auto operator()(T&& a, U&& b) const
        -> decltype(a < b ? a : b) {
        return a < b ? a : b;
    }
};

} // namespace thrust
