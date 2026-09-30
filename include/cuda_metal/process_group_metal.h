#pragma once

// CUDA-to-Metal Translation Layer
// torch.distributed Process Group backend for Metal/NCCL
// Wraps the NCCL communicator (src/nccl/) and exposes the
// collective / P2P interface that PyTorch's ProcessGroup expects.

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace cuda_metal {

// Work handle for async collective operations
class WorkMetal {
public:
    WorkMetal();
    bool isCompleted() const;
    bool isSuccess() const;
    void wait();
    void synchronize();

    // Mark the work as finished with a success/failure status
    void finish(bool success);

private:
    bool completed_ = false;
    bool success_ = false;
};

// Tensor descriptor for distributed ops
struct DistTensor {
    void* data;
    size_t numel;
    size_t element_size;  // bytes per element
    int dtype;            // maps to ncclDataType_t
};

// Process group wrapping NCCL for Metal
class ProcessGroupMetal {
public:
    ProcessGroupMetal(int rank, int world_size);
    ~ProcessGroupMetal();

    int getRank() const;
    int getSize() const;

    // Collectives - return Work handle
    std::shared_ptr<WorkMetal> allreduce(std::vector<DistTensor>& tensors,
                                         int reduce_op = 0 /*ncclSum*/);
    std::shared_ptr<WorkMetal> broadcast(std::vector<DistTensor>& tensors,
                                          int root = 0);
    std::shared_ptr<WorkMetal> allgather(
        std::vector<std::vector<DistTensor>>& outputs,
        std::vector<DistTensor>& inputs);
    std::shared_ptr<WorkMetal> reduce_scatter(
        std::vector<DistTensor>& outputs,
        std::vector<std::vector<DistTensor>>& inputs,
        int reduce_op = 0);
    std::shared_ptr<WorkMetal> barrier();

    // Point-to-point
    std::shared_ptr<WorkMetal> send(std::vector<DistTensor>& tensors, int dst);
    std::shared_ptr<WorkMetal> recv(std::vector<DistTensor>& tensors, int src);

private:
    int rank_;
    int world_size_;
    void* nccl_comm_;     // ncclComm_t stored as opaque pointer
    cudaStream_t stream_; // dedicated stream for collective ops
};

// Factory function -- sets up the NCCL communicator across ranks.
// master_addr / master_port are used by the underlying NCCL bootstrap
// (currently a no-op for single-process / shared-memory mode).
std::shared_ptr<ProcessGroupMetal> createProcessGroupMetal(
    int rank, int world_size,
    const std::string& master_addr = "127.0.0.1",
    int master_port = 29500);

}  // namespace cuda_metal
