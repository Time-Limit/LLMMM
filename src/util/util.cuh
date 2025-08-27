#pragma once

#include "util/util.h"
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdexcept>
#include <type_traits>

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

__device__ __inline__ void print_thread_info(const char* prefix = "")
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

__forceinline__ __device__ void
mma_m16n8k32_row_col(float (&d)[4], const __nv_fp8_e4m3 (&a)[16], const __nv_fp8_e4m3 (&b)[8], const float (&c)[4])
{
  uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
  uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
  float const*    C = reinterpret_cast<float const*>(&c);
  float*          D = reinterpret_cast<float*>(&d);
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"
    "{%0,  %1, %2, %3},"
    "{%4,  %5, %6, %7},"
    "{%8, %9},"
    "{%10, %11, %12, %13};"
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2>>
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

template<typename T, int N, typename = std::enable_if_t<sizeof(T) == 4 and N >= 2>>
__forceinline__ __device__ void shfl_1_and_0(T (&data)[N], uint32_t mask, int lane_id)
{
  uint32_t& _0   = *(uint32_t*)(&data[0]);
  uint32_t& _1   = *(uint32_t*)(&data[1]);
  uint32_t  swap = (_0 ^ _1) * (!(lane_id & mask));
  _0 ^= swap;
  _1 ^= swap;
  _0   = __shfl_xor_sync(0xffffffff, _0, mask);
  swap = (_0 ^ _1) * (!(lane_id & mask));
  _0 ^= swap;
  _1 ^= swap;
}

template<typename T, int N, typename = std::enable_if_t<sizeof(T) == 4 and N >= 4>>
__forceinline__ __device__ void shfl_3_and_2(T (&data)[N], uint32_t mask, int lane_id)
{
  uint32_t& _2   = *(uint32_t*)(&data[2]);
  uint32_t& _3   = *(uint32_t*)(&data[3]);
  uint32_t  swap = (_2 ^ _3) * (!(lane_id & mask));
  _2 ^= swap;
  _3 ^= swap;
  _2   = __shfl_xor_sync(0xffffffff, _2, mask);
  swap = (_2 ^ _3) * (!(lane_id & mask));
  _2 ^= swap;
  _3 ^= swap;
}

template<typename T, int N, typename = std::enable_if_t<(sizeof(T) == 2 || sizeof(T) == 4) && N >= 4>>
__forceinline__ __device__ void shfl_23_and_01(T (&data)[N], uint32_t mask, int lane_id)
{
  if constexpr (sizeof(T) == 2) {
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
  if constexpr (sizeof(T) == 4) {
    uint64_t& _01  = *(uint64_t*)(&data[0]);
    uint64_t& _23  = *(uint64_t*)(&data[2]);
    uint64_t  swap = (_01 ^ _23) * (!(lane_id & mask));
    _01 ^= swap;
    _23 ^= swap;
    _01  = __shfl_xor_sync(0xffffffff, _01, mask);
    swap = (_01 ^ _23) * (!(lane_id & mask));
    _01 ^= swap;
    _23 ^= swap;
  }
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2 || sizeof(T) == 1>>
__forceinline__ __device__ void shfl_4567_and_0123(T (&data)[8], uint32_t mask, int lane_id)
{
  if constexpr (sizeof(T) == 2) {
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
  if constexpr (sizeof(T) == 1) {
    uint32_t& _0123 = *(uint32_t*)(&data[0]);
    uint32_t& _4567 = *(uint32_t*)(&data[4]);
    uint32_t  swap  = (_0123 ^ _4567) * (!(lane_id & mask));
    _0123 ^= swap;
    _4567 ^= swap;
    _0123 = __shfl_xor_sync(0xffffffff, _0123, mask);
    swap  = (_0123 ^ _4567) * (!(lane_id & mask));
    _0123 ^= swap;
    _4567 ^= swap;
  }
}

__forceinline__ __device__ float warp_reduce_max(float val)
{
  val = max(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = max(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = max(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = max(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = max(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

__forceinline__ __device__ float warp_reduce_min(float val)
{
  val = min(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = min(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = min(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = min(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = min(val, __shfl_xor_sync(0xffffffff, val, 1));
  return val;
}

__forceinline__ __device__ float warp_broadcast(int src_lane_id, float val)
{
  return __shfl_sync(0xffffffff, val, src_lane_id);
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2>>
__forceinline__ __device__ void ldmatrix_sync_aligned_m8n8_x2_b16(T (&dst)[2][2], const void* src_ptr)
{
  uint32_t src = __cvta_generic_to_shared(src_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(*(uint32_t*)&dst[0][0]), "=r"(*(uint32_t*)&dst[1][0])
               : "r"(src));
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 2>>
__forceinline__ __device__ void ldmatrix_sync_aligned_m8n8_x4_b16(T (&dst)[4][2], const void* src_ptr)
{
  uint32_t src = __cvta_generic_to_shared(src_ptr);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(*(uint32_t*)&dst[0][0]),
                 "=r"(*(uint32_t*)&dst[1][0]),
                 "=r"(*(uint32_t*)&dst[2][0]),
                 "=r"(*(uint32_t*)&dst[3][0])
               : "r"(src));
}

template<typename T, int N>
__forceinline__ __device__ constexpr int get_array_size(const T (&data)[N])
{
  return N;
}

}  // namespace LLMMM
