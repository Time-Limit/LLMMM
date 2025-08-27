#include "base/base.h"
#include "util/macro.h"
#include "util/util.cuh"
#include <cuda_fp8.h>

namespace LLMMM {

template<typename T, typename = std::enable_if_t<sizeof(T) == 1>>
__global__ void transpose_matrix__aligned_128__T_size_1(T* dst, const T* src, int M, int N)
{
  const int m_warp_count = M / 128;
  // const int n_warp_count  = N / 128;
  const int block_id      = blockIdx.x;
  const int warp_id       = threadIdx.x / 32;
  const int lane_id       = threadIdx.x % 32;
  const int warp_count    = blockDim.x / 32;
  const int m_warp_offset = (block_id * warp_count + warp_id) % m_warp_count * 128;
  const int n_warp_offset = (block_id * warp_count + warp_id) / m_warp_count * 128;
  if (m_warp_offset >= M || n_warp_offset >= N) {
    return;
  }
  T data[2][16][16];
  for (int loop = 0; loop < 32; ++loop) {
    const int m = lane_id % 8 * 16 + loop % 16;
    const int n = lane_id / 8 * 16 + loop / 16 * 64;
    FETCH_FLOAT4(data[loop / 16][loop % 16], src[OFFSET(m_warp_offset + m, n_warp_offset + n, N)]);
  }
  for (int loop = 0; loop < 32; ++loop) {
    const int m              = lane_id / 8 * 16 + loop % 16 + loop / 16 * 64;
    const int n              = lane_id % 8 * 16;
    T         transposed[16] = {
      data[loop / 16][0][loop % 16],
      data[loop / 16][1][loop % 16],
      data[loop / 16][2][loop % 16],
      data[loop / 16][3][loop % 16],
      data[loop / 16][4][loop % 16],
      data[loop / 16][5][loop % 16],
      data[loop / 16][6][loop % 16],
      data[loop / 16][7][loop % 16],
      data[loop / 16][8][loop % 16],
      data[loop / 16][9][loop % 16],
      data[loop / 16][10][loop % 16],
      data[loop / 16][11][loop % 16],
      data[loop / 16][12][loop % 16],
      data[loop / 16][13][loop % 16],
      data[loop / 16][14][loop % 16],
      data[loop / 16][15][loop % 16],
    };
    STORE_FLOAT4(dst[OFFSET(n_warp_offset + m, m_warp_offset + n, M)], transposed);
  }
}

template<typename T, typename = std::enable_if_t<sizeof(T) == 4, T>>
__global__ void transpose_matrix__aligned_128__T_size_4(T* dst, const T* src, int M, int N)
{
  const int m_warp_count = M / 64;
  // const int n_warp_count  = N / 64;
  const int block_id      = blockIdx.x;
  const int warp_id       = threadIdx.x / 32;
  const int lane_id       = threadIdx.x % 32;
  const int warp_count    = blockDim.x / 32;
  const int m_warp_offset = (block_id * warp_count + warp_id) % m_warp_count * 64;
  const int n_warp_offset = (block_id * warp_count + warp_id) / m_warp_count * 64;
  if (m_warp_offset >= M || n_warp_offset >= N) {
    return;
  }
  T data[4][2][4][4];
  for (int m_loop = 0; m_loop < 4; ++m_loop) {
    for (int n_loop = 0; n_loop < 2; ++n_loop) {
      const int m_loop_offset = m_loop * 16;
      const int n_loop_offset = n_loop * 32;
      const int m             = m_warp_offset + m_loop_offset + lane_id / 8 * 4;
      const int n             = n_warp_offset + n_loop_offset + lane_id % 8 * 4;
      FETCH_FLOAT4(data[m_loop][n_loop][0], src[OFFSET(m + 0, n, N)]);
      FETCH_FLOAT4(data[m_loop][n_loop][1], src[OFFSET(m + 1, n, N)]);
      FETCH_FLOAT4(data[m_loop][n_loop][2], src[OFFSET(m + 2, n, N)]);
      FETCH_FLOAT4(data[m_loop][n_loop][3], src[OFFSET(m + 3, n, N)]);
    }
  }
  // if (this_thread_can_log()) {
  //   printf("%8.3f %8.3f %8.3f %8.3f\n", data[0][0][0][0], data[0][0][0][1], data[0][0][0][2], data[0][0][0][3]);
  //   printf("%8.3f %8.3f %8.3f %8.3f\n", data[0][0][1][0], data[0][0][1][1], data[0][0][1][2], data[0][0][1][3]);
  //   printf("%8.3f %8.3f %8.3f %8.3f\n", data[0][0][2][0], data[0][0][2][1], data[0][0][2][2], data[0][0][2][3]);
  //   printf("%8.3f %8.3f %8.3f %8.3f\n", data[0][0][3][0], data[0][0][3][1], data[0][0][3][2], data[0][0][3][3]);
  // }
  for (int m_loop = 0; m_loop < 4; ++m_loop) {
    for (int n_loop = 0; n_loop < 2; ++n_loop) {
      const int m_loop_offset   = n_loop * 32;
      const int n_loop_offset   = m_loop * 16;
      const int m               = n_warp_offset + m_loop_offset + lane_id % 8 * 4;
      const int n               = m_warp_offset + n_loop_offset + lane_id / 8 * 4;
      T         transposed_0[4] = {
        data[m_loop][n_loop][0][0],
        data[m_loop][n_loop][1][0],
        data[m_loop][n_loop][2][0],
        data[m_loop][n_loop][3][0],
      };
      STORE_FLOAT4(dst[OFFSET(m, n, M)], transposed_0);
      T transposed_1[4] = {
        data[m_loop][n_loop][0][1],
        data[m_loop][n_loop][1][1],
        data[m_loop][n_loop][2][1],
        data[m_loop][n_loop][3][1],
      };
      STORE_FLOAT4(dst[OFFSET(m + 1, n, M)], transposed_1);
      T transposed_2[4] = {
        data[m_loop][n_loop][0][2],
        data[m_loop][n_loop][1][2],
        data[m_loop][n_loop][2][2],
        data[m_loop][n_loop][3][2],
      };
      STORE_FLOAT4(dst[OFFSET(m + 2, n, M)], transposed_2);
      T transposed_3[4] = {
        data[m_loop][n_loop][0][3],
        data[m_loop][n_loop][1][3],
        data[m_loop][n_loop][2][3],
        data[m_loop][n_loop][3][3],
      };
      STORE_FLOAT4(dst[OFFSET(m + 3, n, M)], transposed_3);
    }
  }
}

template<typename T>
void transpose_matrix__aligned_128(T* dst, T* src, int M, int N, cudaStream_t stream)
{
  if (M % 128 || N % 128 != 0) {
    throw std::runtime_error("M or N can't be divided 128.");
  }
  if constexpr (sizeof(T) == 1) {
    const int warp_count        = M / 128 * N / 128;
    int       warp_count_per_SM = (warp_count + 127) / 128;
    int       block_count       = 128;
    while (warp_count_per_SM > 8) {
      warp_count_per_SM /= 2;
      block_count *= 2;
    }
    transpose_matrix__aligned_128__T_size_1<T><<<block_count, warp_count_per_SM * 32, 0, stream>>>(dst, src, M, N);
  }
  if constexpr (sizeof(T) == 4) {
    const int warp_count        = M / 64 * N / 64;
    int       warp_count_per_SM = (warp_count + 127) / 128;
    int       block_count       = 128;
    while (warp_count_per_SM > 8) {
      warp_count_per_SM /= 2;
      block_count *= 2;
    }
    transpose_matrix__aligned_128__T_size_4<T><<<block_count, warp_count_per_SM * 32, 0, stream>>>(dst, src, M, N);
  }
  static_assert(sizeof(T) == 1 || sizeof(T) == 4);
}

template void transpose_matrix__aligned_128(__nv_fp8_e4m3* dst, __nv_fp8_e4m3* src, int M, int N, cudaStream_t stream);
template void transpose_matrix__aligned_128(float* dst, float* src, int M, int N, cudaStream_t stream);
}  // namespace LLMMM
