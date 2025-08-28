#include "base/base.h"
#include "util/macro.h"
#include "util/util.cuh"
#include <cuda_fp8.h>
#include <stdexcept>

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

template<int BLOCK_M, int BLOCK_K>
__global__ void construct_m16n8k32_A_layout(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int M, int K)
{
  constexpr int WARP_COUNT = 4;
  static_assert(WARP_COUNT * 32 == BLOCK_K);
  const int lane_id = threadIdx.x % 32;
  const int warp_id = threadIdx.x / 32;

  const int m_block_offset = blockIdx.y * BLOCK_M;
  const int k_block_offset = blockIdx.x * BLOCK_K;
  const int k_warp_offset  = warp_id * 32;
  /* Thread Layout. Each thread reads 16 consecutive fp8 along the K dimension.*/
  /* M ↓, K → */
  /* T00 T02 */
  /* T04 T06 */
  /* T08 T10 */
  /* T12 T14 */
  /* T16 T18 */
  /* T20 T22 */
  /* T24 T26 */
  /* T28 T30 */
  /* T01 T03 */
  /* T05 T07 */
  /* T09 T11 */
  /* T13 T15 */
  /* T17 T19 */
  /* T21 T23 */
  /* T25 T27 */
  /* T29 T31 */
  const int m_lane_offset = lane_id / 4 + (lane_id & 0x1) * 8;
  const int k_lane_offset = lane_id % 4 / 2 * 16;

  float reg[4];

  constexpr int    LANE_GROUP_COUNT = 32 / 4;
  __shared__ float sm[WARP_COUNT][4][LANE_GROUP_COUNT][4];

  for (int m_loop_offset = 0; m_loop_offset < BLOCK_M; m_loop_offset += 16) {
    const int m_global = m_block_offset + m_loop_offset + m_lane_offset;
    const int k_global = k_block_offset + k_warp_offset + k_lane_offset;
    FETCH_FLOAT4(reg, src[OFFSET(m_global, k_global, K)]);
    STORE_FLOAT(sm[warp_id][0][lane_id / 4][lane_id % 4], reg[0]);
    STORE_FLOAT(sm[warp_id][1][lane_id / 4][lane_id % 4], reg[1]);
    STORE_FLOAT(sm[warp_id][2][lane_id / 4][lane_id % 4], reg[2]);
    STORE_FLOAT(sm[warp_id][3][lane_id / 4][lane_id % 4], reg[3]);
    FETCH_FLOAT4(reg, sm[warp_id][lane_id % 4][lane_id / 4]);
    STORE_FLOAT4(dst[OFFSET(m_global, k_global, K)], reg);
  }
}

void construct_m16n8k32_A_layout(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int M, int K, cudaStream_t stream)
{
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_K = 128;
  if (K % BLOCK_K != 0) {
    throw std::runtime_error("K must be greater than 0 and divisible by 128.");
  }
  if (M % 16 != 0) {
    throw std::runtime_error("M must be greater than 0 and divisible by 16.");
  }
  dim3 grid(K / BLOCK_K, (M + BLOCK_M - 1) / BLOCK_M);
  dim3 block(128);
  construct_m16n8k32_A_layout<BLOCK_M, BLOCK_K><<<grid, block, 0, stream>>>(dst, src, M, K);
}

template<int BLOCK_N, int BLOCK_K>
__global__ void construct_m16n8k32_B_layout(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src_tranposed, int N, int K)
{
  constexpr int WARP_COUNT = 4;
  static_assert(WARP_COUNT * 32 == BLOCK_K);
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  const int n_block_offset = blockIdx.x * BLOCK_N;
  const int k_block_offset = blockIdx.y * BLOCK_K;
  const int k_warp_offset  = warp_id * 32;
  const int n_lane_offset  = lane_id / 4;
  const int k_lane_offset  = lane_id % 4 * 8;

  /* Thread Layout. Each thread reads 8 consecutive fp8 along the K dimension.*/
  /* N ↓, K → */
  /* T00 T01 T02 T03 */
  /* T04 T05 T06 T07 */
  /* T08 T09 T10 T11 */
  /* T12 T13 T14 T15 */
  /* T16 T17 T18 T19 */
  /* T20 T21 T22 T23 */
  /* T24 T25 T26 T27 */
  /* T28 T29 T30 T31 */
  float            reg[2];
  __shared__ float sm[WARP_COUNT][8][8];
  for (int n_loop_offset = 0; n_loop_offset < BLOCK_N; n_loop_offset += 8) {
    const int n_global = n_block_offset + n_loop_offset + n_lane_offset;
    const int k_global = k_block_offset + k_warp_offset + k_lane_offset;
    if (n_global < N) {
      FETCH_FLOAT2(reg, src_tranposed[OFFSET(n_global, k_global, K)]);
      STORE_FLOAT2(sm[warp_id][lane_id / 4][lane_id % 4 * 2], reg);
      constexpr int position_mapping[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      FETCH_FLOAT(reg[0], sm[warp_id][lane_id / 4][position_mapping[lane_id % 4 * 2]]);
      FETCH_FLOAT(reg[1], sm[warp_id][lane_id / 4][position_mapping[lane_id % 4 * 2 + 1]]);
      STORE_FLOAT2(dst[OFFSET(n_global, k_global, K)], reg);
    }
  }
}

void construct_m16n8k32_B_layout(
  __nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src_tranposed, int N, int K, cudaStream_t stream)
{
  constexpr int BLOCK_K = 128;
  constexpr int BLOCK_N = 128;
  if (K % BLOCK_K != 0) {
    throw std::runtime_error("K must be greater than 0 and divisible by 128.");
  }
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, K / BLOCK_K);
  dim3 block(128);
  construct_m16n8k32_B_layout<BLOCK_N, BLOCK_K><<<grid, block, 0, stream>>>(dst, src_tranposed, N, K);
}

}  // namespace LLMMM
