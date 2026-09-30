#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace cuda_metal {

// Dynamic batching for variable-length inference requests.
//
// Accumulates requests, groups by similar sequence length to minimize
// padding waste, then dispatches as padded batches.

struct BatchRequest {
    const void* input;       // Pointer to token IDs or embeddings
    size_t      seq_len;     // Sequence length of this request
    size_t      request_id;  // Caller-assigned ID for result routing
};

struct BatchGroup {
    std::vector<BatchRequest> requests;
    size_t padded_seq_len;   // Max seq_len in this group (all padded to this)
    size_t batch_size;       // Number of requests in this group
};

// Callback invoked when a batch group is ready for dispatch.
using BatchDispatchFn = std::function<void(const BatchGroup&)>;

class DynamicBatcher {
public:
    static DynamicBatcher& instance();

    // Configure batching parameters.
    //   max_batch:  Maximum requests per batch group
    //   max_tokens: Maximum total tokens (batch_size * padded_seq_len) per group
    //   bin_width:  Sequence lengths within this range get grouped together
    void configure(size_t max_batch = 32, size_t max_tokens = 4096, size_t bin_width = 64);

    // Set the dispatch callback.
    void setDispatchFn(BatchDispatchFn fn);

    // Submit a request. May trigger dispatch if batch is full.
    void submit(const BatchRequest& req);

    // Force-flush all pending requests as batch groups.
    void flush();

    // Number of pending (undelivered) requests.
    size_t pendingCount() const;

    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

private:
    DynamicBatcher() = default;

    size_t max_batch_  = 32;
    size_t max_tokens_ = 4096;
    size_t bin_width_  = 64;

    BatchDispatchFn dispatch_fn_;
    std::vector<BatchRequest> pending_;
    mutable std::mutex mutex_;

    // Form batch groups from pending requests and dispatch them.
    void formAndDispatch();

    // Bin a sequence length to its group key.
    size_t binKey(size_t seq_len) const {
        return ((seq_len + bin_width_ - 1) / bin_width_) * bin_width_;
    }
};

} // namespace cuda_metal
