#pragma once

#include <cuda_runtime.h>

namespace LLMMM {

template<typename T>
void transpose_matrix__aligned_128(T* dst, T* src, int M, int N, cudaStream_t stream);

}
