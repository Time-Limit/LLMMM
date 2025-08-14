#pragma once

#include "mm_impl.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace LLMMM {

// TODO  migrate from LeetKernel
template<>
class MM_IMPL<float> {
public:
  void operator()(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)
  {
    throw std::runtime_error("");
  }

  void tune(int N, int K)
  {
    throw std::runtime_error("");
  }

  void verify(const float* A, const float* B, const float* benchmark, int M, int N, int K)
  {
    throw std::runtime_error("");
  }
};

}  // namespace LLMMM
