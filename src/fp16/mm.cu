#include "fp16/impl.h"
#include "fp16/mm.cuh"

namespace LLMMM {

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_M_SPLIT_COUNT, int BLOCK_N_SPLIT_COUNT, int WARP_M, int WARP_N>
void* get_launcher()
{
  auto func = launch_fp16_mm_kernel<T, BLOCK_M, BLOCK_N, BLOCK_M_SPLIT_COUNT, BLOCK_N_SPLIT_COUNT, WARP_M, WARP_N>;
  return reinterpret_cast<void*>(
    &launch_fp16_mm_kernel<T, BLOCK_M, BLOCK_N, BLOCK_M_SPLIT_COUNT, BLOCK_N_SPLIT_COUNT, WARP_M, WARP_N>);
}

template<typename T>
void FP16_MM_IMPL<T>::init_kernels()
{
  constexpr int BLOCK_M             = 256;
  constexpr int BLOCK_N             = 256;
  constexpr int BLOCK_M_SPLIT_COUNT = 2;
  constexpr int BLOCK_N_SPLIT_COUNT = 2;
  constexpr int WARP_M              = 64;
  constexpr int WARP_N              = 64;
  auto func = launch_fp16_mm_kernel<T, BLOCK_M, BLOCK_N, BLOCK_M_SPLIT_COUNT, BLOCK_N_SPLIT_COUNT, WARP_M, WARP_N>;
  typename FP16_MM_IMPL<T>::Kernel kernel{
    .BLOCK_M             = BLOCK_M,
    .BLOCK_N             = BLOCK_N,
    .BLOCK_M_SPLIT_COUNT = BLOCK_M_SPLIT_COUNT,
    .BLOCK_N_SPLIT_COUNT = BLOCK_N_SPLIT_COUNT,
    .WARP_M              = WARP_M,
    .WARP_N              = WARP_N,
    .launcher            = reinterpret_cast<typename FP16_MM_IMPL<T>::Kernel::launcher_type>(
      get_launcher<T, BLOCK_M, BLOCK_N, BLOCK_M_SPLIT_COUNT, BLOCK_N_SPLIT_COUNT, WARP_M, WARP_N>())};
  printf(
    "ADD FP16_MM_KERNEL, BLOCK_M=%03d, BLOCK_N=%03d, BLOCK_M_SPLIT_COUNT=%03d, BLOCK_N_SPLIT_COUNT=%03d, WARP_M=%03d,WARP_N=%03d, launcher = %p\n",
    BLOCK_M,
    BLOCK_N,
    BLOCK_M_SPLIT_COUNT,
    BLOCK_N_SPLIT_COUNT,
    WARP_M,
    WARP_N,
    kernel.launcher);
  FP16_MM_IMPL<T>::kernels.emplace_back(kernel);
}

template class FP16_MM_IMPL<half>;
template class FP16_MM_IMPL<__nv_bfloat16>;

}  // namespace LLMMM
