#pragma once

#include "fp16/impl.h"
#include "fp32/impl.h"

namespace LLMMM {

template<typename T>
class MM {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>);

  MM_IMPL<T> mm_impl;

public:
  MM()                     = default;
  ~MM()                    = default;
  MM(const MM&)            = delete;
  MM operator=(const MM&)  = delete;
  MM(MM&&)                 = delete;
  MM operator=(const MM&&) = delete;

  void operator()(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream)
  {
    mm_impl(A, B, C, M, N, K, stream);
  }

  void tune(int N, int K)
  {
    mm_impl.tune(N, K);
  }

  void verify(const float* A, const float* B, const float* benchmark, int M, int N, int K)
  {
    mm_impl.verify(A, B, benchmark, M, N, K);
  }
};

}  // namespace LLMMM
