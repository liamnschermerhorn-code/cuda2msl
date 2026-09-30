#pragma once

// CUDA-to-Metal Translation Layer
// Drop-in replacement for NVIDIA's nccl.h
// On Apple Silicon with unified memory, NCCL collectives become
// shared-memory operations (or no-ops for single-rank).

#include "cuda_runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

#define NCCL_MAJOR   2
#define NCCL_MINOR   19
#define NCCL_PATCH   1
#define NCCL_VERSION_CODE  21901

// ---------------------------------------------------------------------------
// Result codes
// ---------------------------------------------------------------------------

typedef enum {
    ncclSuccess            = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError        = 2,
    ncclInternalError      = 3,
    ncclInvalidArgument    = 4,
    ncclInvalidUsage       = 5
} ncclResult_t;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

typedef enum {
    ncclInt8    = 0,
    ncclChar    = 0,
    ncclUint8   = 1,
    ncclInt32   = 2,
    ncclInt     = 2,
    ncclUint32  = 3,
    ncclInt64   = 4,
    ncclUint64  = 5,
    ncclFloat16 = 6,
    ncclHalf    = 6,
    ncclFloat32 = 7,
    ncclFloat   = 7,
    ncclFloat64 = 8,
    ncclDouble  = 8,
    ncclBfloat16 = 9
} ncclDataType_t;

// ---------------------------------------------------------------------------
// Reduction operations
// ---------------------------------------------------------------------------

typedef enum {
    ncclSum  = 0,
    ncclProd = 1,
    ncclMax  = 2,
    ncclMin  = 3,
    ncclAvg  = 4
} ncclRedOp_t;

// ---------------------------------------------------------------------------
// Opaque types
// ---------------------------------------------------------------------------

typedef struct ncclComm* ncclComm_t;

typedef struct {
    char internal[128];
} ncclUniqueId;

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

ncclResult_t ncclGetVersion(int* version);
const char*  ncclGetErrorString(ncclResult_t result);

ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);

// Initialize a communicator for rank `rank` out of `nranks` total.
// Uses the uniqueId to coordinate across processes.
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                              ncclUniqueId commId, int rank);

// Initialize all `ndev` communicators in a single process.
// comms[] must have room for ndev entries.
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev,
                             const int* devlist);

// ---------------------------------------------------------------------------
// Communicator queries
// ---------------------------------------------------------------------------

ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);

// ---------------------------------------------------------------------------
// Collective operations
// ---------------------------------------------------------------------------

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff,
                           size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream);

ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
                           size_t count, ncclDataType_t datatype,
                           int root, ncclComm_t comm,
                           cudaStream_t stream);

ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff,
                        size_t count, ncclDataType_t datatype,
                        ncclRedOp_t op, int root, ncclComm_t comm,
                        cudaStream_t stream);

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream);

// ---------------------------------------------------------------------------
// Point-to-point operations
// ---------------------------------------------------------------------------

ncclResult_t ncclSend(const void* sendbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclRecv(void* recvbuff, size_t count,
                      ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Group semantics (batch multiple collectives)
// ---------------------------------------------------------------------------

ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

#ifdef __cplusplus
}
#endif
