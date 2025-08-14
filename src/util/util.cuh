#pragma once

#include "util/util.h"
#include <cstdio>
#include <cuda_fp16.h>

namespace LLMMM {

__device__ __inline__ bool this_block_can_log()
{
  return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

__device__ __inline__ bool this_thread_can_log(int thread_x = -1)
{
  if (thread_x == -1) {
    return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0
           && threadIdx.z == 0;
  }
  return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == thread_x && threadIdx.y == 0
         && threadIdx.z == 0;
}

__device__ __inline__ void print_thread_info(const char* prefix)
{
  printf("%s, block = %03d, %03d, %03d, thread = %03d %03d %03d\n",
         prefix,
         blockIdx.x,
         blockIdx.y,
         blockIdx.z,
         threadIdx.x,
         threadIdx.y,
         threadIdx.z);
}

template<int BLOCK_TILE_M, int BLOCK_TILE_N, int THREAD_TILE_M, int THREAD_TILE_N>
__device__ __inline__ constexpr int device_thread_count_calculator()
{
  thread_count_calculator();
}

template<int UNALIGNED_BLOCK_TILE_M>
__device__ __inline__ constexpr int device_corresponding_aligned_M_calculator()
{
  corresponding_aligned_M_calculator();
}

template<typename T>
__inline__ __device__ void
mma_sync_aligned_m8n8k4_row_row_f32_f16_f16_f32(float (&D)[8], const T (&A)[4], const T (&B)[4], const float (&C)[8])
{
  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32"
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
               "{%8,  %9},"
               "{%10, %11},"
               "{%12, %13, %14, %15, %16, %17, %18, %19};"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[4]), "=f"(D[5]), "=f"(D[2]), "=f"(D[3]), "=f"(D[6]), "=f"(D[7])
               : "r"(*(uint32_t*)&A[0]),
                 "r"(*(uint32_t*)&A[2]),
                 "r"(*(uint32_t*)&B[0]),
                 "r"(*(uint32_t*)&B[2]),
                 "f"(C[0]),
                 "f"(C[1]),
                 "f"(C[4]),
                 "f"(C[5]),
                 "f"(C[2]),
                 "f"(C[3]),
                 "f"(C[6]),
                 "f"(C[7]));
}

template<typename T>
__forceinline__ __device__ void
mma_m16n8k16_row_col(float (&d)[4], const T (&a)[8], const T (&b)[4], const float (&c)[4])
{
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
  float const*    C = reinterpret_cast<float const*>(&c);
  float*          D = reinterpret_cast<float*>(&d);
  if constexpr (std::is_same<T, half>::value) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
  }
  else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
  }
  else {
    static_assert(std::is_same<T, half>::value == false && std::is_same<T, __nv_bfloat16>::value == false);
  }
}

__forceinline__ __device__ uint32_t get_smid()
{
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__forceinline__ uint32_t get_sm_number()
{
  uint32_t sm_number;
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(sm_number));
  return sm_number;
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2>>
__forceinline__ __device__ void shfl_23_and_01(T (&data)[4], uint32_t mask, int lane_id)
{
  uint32_t& _01  = *(uint32_t*)(&data[0]);
  uint32_t& _23  = *(uint32_t*)(&data[2]);
  uint32_t  swap = (_01 ^ _23) * (!(lane_id & mask));
  _01 ^= swap;
  _23 ^= swap;
  _01  = __shfl_xor_sync(0xffffffff, _01, mask);
  swap = (_01 ^ _23) * (!(lane_id & mask));
  _01 ^= swap;
  _23 ^= swap;
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2>>
__forceinline__ __device__ void shfl_4567_and_0123(T (&data)[8], uint32_t mask, int lane_id)
{
  uint64_t& _0123 = *(uint64_t*)(&data[0]);
  uint64_t& _4567 = *(uint64_t*)(&data[4]);
  uint64_t  swap  = (_0123 ^ _4567) * (!(lane_id & mask));
  _0123 ^= swap;
  _4567 ^= swap;
  _0123 = __shfl_xor_sync(0xffffffff, _0123, mask);
  swap  = (_0123 ^ _4567) * (!(lane_id & mask));
  _0123 ^= swap;
  _4567 ^= swap;
}

}  // namespace LLMMM
