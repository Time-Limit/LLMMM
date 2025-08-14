#pragma once

#include <cuda_runtime.h>

namespace LLMMM {

template<typename T>
class MM_IMPL {
public:
  void operator()(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream);

  void tune(int N, int K);

  void verify(const float* A, const float* B, const float* benchmark, int M, int N, int K);
};

}  // namespace LLMMM
