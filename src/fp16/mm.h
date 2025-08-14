#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace LLMMM {

template<typename T>
void fp16_mm(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream);

}  // namespace LLMMM
