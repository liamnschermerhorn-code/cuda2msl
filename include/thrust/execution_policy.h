#pragma once

// Thrust-compatible execution policy tags for CUDA-to-Metal translation layer.
// Since Metal unified memory makes device pointers host-accessible, both
// policies dispatch to the same CPU-side STL implementations.

namespace thrust {

struct device_execution_policy {};
struct host_execution_policy {};

// Global policy tag instances
static constexpr device_execution_policy device{};
static constexpr host_execution_policy   host{};

// Execution system tags used by some Thrust internals
namespace system {
namespace detail {
struct sequential_tag {};
} // namespace detail
} // namespace system

} // namespace thrust
