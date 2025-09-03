#include "mm/fp16/mm.cuh"
#include "mm/fp16/mm.h"

namespace LLMMM {

template<typename T>
void fp16_mm(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream)
{
  constexpr int BLOCK_M             = 256;
  constexpr int BLOCK_N             = 256;
  constexpr int BLOCK_M_SPLIT_COUNT = 2;
  constexpr int BLOCK_N_SPLIT_COUNT = 2;
  constexpr int WARP_M              = 64;
  constexpr int WARP_N              = 64;
  if (std::is_same<T, half>::value == false && std::is_same<T, __nv_bfloat16>::value == false) {
    throw std::runtime_error("T is not supported.");
  }
  constexpr int LOOP_K = 16;
  if (!(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % LOOP_K == 0)) {
    throw std::runtime_error("M or N or K are not aligned.");
  }
  thread_local auto set_attr_result = []() {
    auto kSmemSize   = 0;
    auto kernel_func = &fp16_mm<T, BLOCK_M, BLOCK_N, BLOCK_M_SPLIT_COUNT, BLOCK_N_SPLIT_COUNT, WARP_M, WARP_N>;
    CHECK_CUDA_RETURN(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
    return true;
  }();
  static_assert(8 <= BLOCK_M && (BLOCK_M & (BLOCK_M - 1)) == 0);
  static_assert(16 <= BLOCK_N && (BLOCK_N & (BLOCK_N - 1)) == 0);
  static_assert(LOOP_K == 16);
  static_assert(BLOCK_M % BLOCK_M_SPLIT_COUNT == 0 && BLOCK_N % BLOCK_N_SPLIT_COUNT == 0);
  constexpr int SPLITED_BLOCK_M = BLOCK_M / BLOCK_M_SPLIT_COUNT;
  constexpr int SPLITED_BLOCK_N = BLOCK_N / BLOCK_N_SPLIT_COUNT;
  static_assert(SPLITED_BLOCK_M % WARP_M == 0 && SPLITED_BLOCK_N % WARP_N == 0);
  static_assert(WARP_N % 16 == 0 && WARP_M % 8 == 0);
  constexpr int WARP_COUNT = SPLITED_BLOCK_N / WARP_N * SPLITED_BLOCK_M / WARP_M;
  static_assert(1 <= WARP_COUNT && WARP_COUNT <= 32 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);
  dim3 grid(N / BLOCK_N, M / BLOCK_M);
  dim3 block(WARP_COUNT * 32);
  fp16_mm<T, BLOCK_M, BLOCK_N, BLOCK_M_SPLIT_COUNT, BLOCK_N_SPLIT_COUNT, WARP_M, WARP_N>
    <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

template void fp16_mm(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t stream);
template void
fp16_mm(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K, cudaStream_t stream);

}  // namespace LLMMM
