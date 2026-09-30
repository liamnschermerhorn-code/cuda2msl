#pragma once

#include "cuda.h"

namespace cuda_metal {

// Thread-local last error tracking
void setLastError(cudaError_t error);
cudaError_t getLastError();
cudaError_t peekLastError();

} // namespace cuda_metal
