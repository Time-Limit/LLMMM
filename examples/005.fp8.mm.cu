#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <limits>
#include <random>
#include <vector>

#include "base/base.h"
#include "util/macro.h"
#include "util/util.cuh"

using namespace LLMMM;

constexpr int limit = 16;

__global__ void fp32_naive_mm(const float* A, const float* B, float* C, int M, int N, int K)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;

  if (n >= N || m >= M) {
    return;
  }

  A += m * K;
  B += n;
  double sum = 0.0;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    sum += A[k] * B[k * N];
  }
  C[m * N + n] = sum;
}

void launch_fp32_naive_mm(const float* A, const float* B, float* C, int M, int N, int K)
{
  dim3 threads_per_block(16, 16);
  dim3 blocks_per_grid((N + threads_per_block.x - 1) / threads_per_block.x,
                       (M + threads_per_block.y - 1) / threads_per_block.y);

  fp32_naive_mm<<<blocks_per_grid, threads_per_block>>>(A, B, C, M, N, K);
  CHECK_CUDA_ERROR();
}

template<int BLOCK_M, int BLOCK_N, int WARP_M, int WARP_N, int LOOP_K>
__global__ void fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32(const __nv_fp8_e4m3* A,
                                                                                  const float* A_scale_transposed,
                                                                                  const __nv_fp8_e4m3* B_transposed,
                                                                                  const float* B_scale_transposed,
                                                                                  float*       C,
                                                                                  int          M,
                                                                                  int          N,
                                                                                  int          K)
{
  constexpr int M_WARP_COUNT     = BLOCK_M / WARP_M;
  constexpr int N_WARP_COUNT     = BLOCK_N / WARP_N;
  constexpr int WARP_COUNT       = M_WARP_COUNT * N_WARP_COUNT;
  constexpr int THREAD_COUNT     = WARP_COUNT * 32;
  constexpr int M_GROUP_PER_WARP = WARP_M / 8;
  constexpr int N_GROUP_PER_WARP = WARP_N / 16;

  using fp8_t = __nv_fp8_e4m3;

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = BLOCK_M * blockIdx.y;
  const int n_block_offset = BLOCK_N * blockIdx.x;
  const int m_warp_offset  = warp_id % M_WARP_COUNT * WARP_M;
  const int n_warp_offset  = warp_id / M_WARP_COUNT * WARP_N;

  // LDG
  __shared__ fp8_t A_sm[LOOP_K / 16][BLOCK_M][16];
  float            A_scale_reg[M_GROUP_PER_WARP][2];  // A_scale_transposed is (K/128) x M
  __shared__ fp8_t B_sm[LOOP_K / 16][BLOCK_N][16];
  float            B_scale_reg;  // B_scale_transposed is (N/128) x (K/128)

  // constexpr int M_BYTE_PER_THREAD = BLOCK_M * LOOP_K / THREAD_COUNT;
  // static_assert(M_BYTE_PER_THREAD % 16 == 0);
  // constexpr int N_BYTE_PER_THREAD = BLOCK_N * LOOP_K / THREAD_COUNT;
  // static_assert(N_BYTE_PER_THREAD % 16 == 0);

  // PARTIAL_LOOP_K is aimed at reducing register usage.
  constexpr int PARTIAL_LOOP_K = 64;
  static_assert(LOOP_K == PARTIAL_LOOP_K * 2);
  constexpr int M_BYTE_PER_THREAD = BLOCK_M * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(M_BYTE_PER_THREAD % 16 == 0);
  constexpr int N_BYTE_PER_THREAD = BLOCK_N * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(N_BYTE_PER_THREAD % 16 == 0);

  constexpr int M_LDG_PER_THREAD = M_BYTE_PER_THREAD / 16;
  constexpr int N_LDG_PER_THREAD = N_BYTE_PER_THREAD / 16;

  fp8_t A_ldg_reg[M_LDG_PER_THREAD][16];
  fp8_t B_ldg_reg[N_LDG_PER_THREAD][16];

  // MMA
  union {
    uint16_t ldm[2][2];
    fp8_t    mma[8];
  } A_cal_reg[M_GROUP_PER_WARP];
  union {
    uint16_t ldm[4][2];
    fp8_t    mma[16];
  } B_cal_reg[N_GROUP_PER_WARP];

  float C_cal_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4] = {0};

  for (int k_block_offset = 0; k_block_offset < K; k_block_offset += 128) {
    for (int k_group_offset = 0; k_group_offset < LOOP_K; k_group_offset += 64) {
      for (int m_loop = 0; m_loop < M_LDG_PER_THREAD; ++m_loop) {
        /* T00 T08 ... T24 */
        /* T01 T09 ... T25 */
        /* T02 T10 ... T26 */
        /* T03 T11 ... T27 */
        /* T04 T12 ... T28 */
        /* T05 T13 ... T29 */
        /* T06 T14 ... T30 */
        /* T07 T15 ... T31 */
        const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 16 / 64 % BLOCK_M;
        const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 16 / 64 / BLOCK_M * 64;
        const int m_lane_offset = lane_id % 8;
        const int k_lane_offset = lane_id / 8 * 16;
        const int m_global      = m_block_offset + m_loop_offset + m_lane_offset;
        const int k_global      = k_block_offset + k_group_offset + k_loop_offset + k_lane_offset;
        FETCH_FLOAT4(A_ldg_reg[m_loop], A[OFFSET(m_global, k_global, K)]);
      }
      for (int m_loop = 0; m_loop < M_LDG_PER_THREAD; ++m_loop) {
        const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 16 / 64 % BLOCK_M;
        const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 16 / 64 / BLOCK_M * 64;
        const int m_lane_offset = lane_id % 8;
        const int k_lane_offset = lane_id / 8 * 16;
        const int m_sm          = m_loop_offset + m_lane_offset;
        const int k_sm          = k_loop_offset + k_group_offset + k_lane_offset;
        STORE_FLOAT4(A_sm[k_sm / 16][m_sm][k_sm % 16], A_ldg_reg[m_loop]);
      }
      for (int n_loop = 0; n_loop < N_LDG_PER_THREAD; ++n_loop) {
        /* T00 T08 ... T24 */
        /* T01 T09 ... T25 */
        /* T02 T10 ... T26 */
        /* T03 T11 ... T27 */
        /* T04 T12 ... T28 */
        /* T05 T13 ... T29 */
        /* T06 T14 ... T30 */
        /* T07 T15 ... T31 */
        const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 % BLOCK_N;
        const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 / BLOCK_N * 64;
        const int n_lane_offset = lane_id % 8;
        const int k_lane_offset = lane_id / 8 * 16;
        const int n_global      = n_block_offset + n_loop_offset + n_lane_offset;
        const int k_global      = k_block_offset + k_group_offset + k_loop_offset + k_lane_offset;
        FETCH_FLOAT4(B_ldg_reg[n_loop], B_transposed[OFFSET(n_global, k_global, K)]);
      }
      for (int n_loop = 0; n_loop < N_LDG_PER_THREAD; ++n_loop) {
        const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 % BLOCK_N;
        const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 / BLOCK_N * 64;
        const int n_lane_offset = lane_id % 8;
        const int k_lane_offset = lane_id / 8 * 16;
        const int n_sm          = n_loop_offset + n_lane_offset;
        const int k_sm          = k_loop_offset + k_group_offset + k_lane_offset;
        STORE_FLOAT4(B_sm[k_sm / 16][n_sm][k_sm % 16], B_ldg_reg[n_loop]);
      }
      FETCH_FLOAT(B_scale_reg, B_scale_transposed[OFFSET(n_block_offset / 128, k_block_offset / 128, K / 128)]);
      for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
        FETCH_FLOAT2(A_scale_reg[mg],
                     A_scale_transposed[OFFSET(
                       k_block_offset / 128, m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2, M)]);
      }
    }
    __syncthreads();
    for (int k_group_offset = 0; k_group_offset < LOOP_K; k_group_offset += 32) {
      for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
        /* 8 x 32 per group, ldmatrix.m8n8.x2.b16 */
        int m_group_offset = m_warp_offset + mg * 8;
        ldmatrix_sync_aligned_m8n8_x2_b16(
          A_cal_reg[mg].ldm,
          &A_sm[k_group_offset / 16 + lane_id / 8][m_group_offset + lane_id % 8][k_group_offset % 16]);
      }
      for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
        /* 16 x 32 per group, ldmatrix.m8n8.x4.b16 */
        int n_group_offset = n_warp_offset + ng * 16;
        ldmatrix_sync_aligned_m8n8_x4_b16(B_cal_reg[ng].ldm,
                                          &B_sm[k_group_offset / 16 + lane_id / 16][n_group_offset + lane_id % 16][0]);
      }
      for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
        for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
          float C_partial_mma_reg[4] = {0};
          mma_m16n8k32_row_col(C_partial_mma_reg, B_cal_reg[ng].mma, A_cal_reg[mg].mma, C_partial_mma_reg);
          C_cal_reg[mg][ng][0] += C_partial_mma_reg[0] * A_scale_reg[mg][0] * B_scale_reg;
          C_cal_reg[mg][ng][1] += C_partial_mma_reg[1] * A_scale_reg[mg][1] * B_scale_reg;
          C_cal_reg[mg][ng][2] += C_partial_mma_reg[2] * A_scale_reg[mg][0] * B_scale_reg;
          C_cal_reg[mg][ng][3] += C_partial_mma_reg[3] * A_scale_reg[mg][1] * B_scale_reg;
        }
      }
    }
    __syncthreads();
  }
  constexpr int m_lane_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  constexpr int n_lane_offset[4] = {0, 8, 4, 12};
  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      LLMMM::shfl_1_and_0(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_3_and_2(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_23_and_01(C_cal_reg[mg][ng], 0x8, lane_id);
      const int m_global = m_block_offset + m_warp_offset + mg * 8 + m_lane_offset[lane_id % 8];
      const int n_global = n_block_offset + n_warp_offset + ng * 16 + n_lane_offset[lane_id / 8];
      STORE_FLOAT4(C[OFFSET(m_global, n_global, N)], C_cal_reg[mg][ng]);
    }
  }
}

void fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32(const __nv_fp8_e4m3* A,
                                                                       const float*         A_scale_transposed,
                                                                       const __nv_fp8_e4m3* B_transposed,
                                                                       const float*         B_scale_transposed,
                                                                       float*               C,
                                                                       int                  M,
                                                                       int                  N,
                                                                       int                  K,
                                                                       cudaStream_t         stream)
{
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 128;
  constexpr int LOOP_K  = 128;
  constexpr int WARP_M  = 64;
  constexpr int WARP_N  = 64;
  static_assert(BLOCK_M > 0 && BLOCK_M <= 128 && BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N == 128 && BLOCK_N % WARP_N == 0);
  static_assert(WARP_M > 0 && WARP_M % 8 == 0);  // mma.m16n8k32.row.col, A is n8k32, B is m16k32
  static_assert(WARP_N > 0 && WARP_N % 16 == 0);
  static_assert(LOOP_K == 128);
  constexpr int WARP_COUNT = BLOCK_M / WARP_M * BLOCK_N / WARP_N;
  static_assert(0 < WARP_COUNT && WARP_COUNT <= 4 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);
  if (!(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % LOOP_K == 0)) {
    throw std::runtime_error("M or N or K are not aligned.");
  }
  dim3 grid(N / BLOCK_N, M / BLOCK_M);
  dim3 block(WARP_COUNT * 32);
  fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32<BLOCK_M, BLOCK_N, WARP_M, WARP_N, LOOP_K>
    <<<grid, block, 0, stream>>>(A, A_scale_transposed, B_transposed, B_scale_transposed, C, M, N, K);
}

template<int BLOCK_M, int BLOCK_N, int WARP_M, int WARP_N, int LOOP_K>
__global__ void
fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32__partial_loop_k(const __nv_fp8_e4m3* A,
                                                                                  const float* A_scale_transposed,
                                                                                  const __nv_fp8_e4m3* B_transposed,
                                                                                  const float* B_scale_transposed,
                                                                                  float*       C,
                                                                                  int          M,
                                                                                  int          N,
                                                                                  int          K)
{
  constexpr int M_WARP_COUNT     = BLOCK_M / WARP_M;
  constexpr int N_WARP_COUNT     = BLOCK_N / WARP_N;
  constexpr int WARP_COUNT       = M_WARP_COUNT * N_WARP_COUNT;
  constexpr int THREAD_COUNT     = WARP_COUNT * 32;
  constexpr int M_GROUP_PER_WARP = WARP_M / 8;
  constexpr int N_GROUP_PER_WARP = WARP_N / 16;

  using fp8_t = __nv_fp8_e4m3;

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = BLOCK_M * blockIdx.y;
  const int n_block_offset = BLOCK_N * blockIdx.x;
  const int m_warp_offset  = warp_id % M_WARP_COUNT * WARP_M;
  const int n_warp_offset  = warp_id / M_WARP_COUNT * WARP_N;

  // LDG
  constexpr int PARTIAL_LOOP_K = 32;
  static_assert(PARTIAL_LOOP_K == 32 && LOOP_K == PARTIAL_LOOP_K * 4);
  __shared__ fp8_t A_sm[PARTIAL_LOOP_K / 16][BLOCK_M][16];
  float            A_scale_reg[M_GROUP_PER_WARP][2];  // A_scale_transposed is (K/128) x M
  __shared__ fp8_t B_sm[PARTIAL_LOOP_K / 16][BLOCK_N][16];
  float            B_scale_reg;  // B_scale_transposed is (N/128) x (K/128)

  // PARTIAL_LOOP_K is aimed at reducing register usage.
  constexpr int BYTE_PER_LDG      = sizeof(float2);
  constexpr int M_BYTE_PER_THREAD = BLOCK_M * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(M_BYTE_PER_THREAD % BYTE_PER_LDG == 0);
  constexpr int N_BYTE_PER_THREAD = BLOCK_N * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(N_BYTE_PER_THREAD % BYTE_PER_LDG == 0);

  constexpr int M_LDG_PER_THREAD = M_BYTE_PER_THREAD / BYTE_PER_LDG;
  constexpr int N_LDG_PER_THREAD = N_BYTE_PER_THREAD / BYTE_PER_LDG;

  fp8_t A_ldg_reg[M_LDG_PER_THREAD][BYTE_PER_LDG];
  fp8_t B_ldg_reg[N_LDG_PER_THREAD][BYTE_PER_LDG];

  // MMA
  union {
    uint16_t ldm[2][2];
    fp8_t    mma[8];
  } A_cal_reg[M_GROUP_PER_WARP];
  union {
    uint16_t ldm[4][2];
    fp8_t    mma[16];
  } B_cal_reg[N_GROUP_PER_WARP];

  float C_cal_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4] = {0};

  for (int k_block_offset = 0; k_block_offset < K; k_block_offset += PARTIAL_LOOP_K) {
    for (int m_loop = 0; m_loop < M_LDG_PER_THREAD; ++m_loop) {
      /* T00 T01 T16 T17 */
      /* T02 T03 T18 T19 */
      /* T04 T05 T20 T21 */
      /* T06 T07 T22 T23 */
      /* T08 T09 T24 T25 */
      /* T10 T11 T26 T27 */
      /* T12 T13 T28 T29 */
      /* T14 T15 T30 T31 */
      const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 % BLOCK_M;
      const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 / BLOCK_M * 32;
      const int m_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int m_global      = m_block_offset + m_loop_offset + m_lane_offset;
      const int k_global      = k_block_offset + k_loop_offset + k_lane_offset;
      FETCH_FLOAT2(A_ldg_reg[m_loop], A[OFFSET(m_global, k_global, K)]);
    }
    for (int n_loop = 0; n_loop < N_LDG_PER_THREAD; ++n_loop) {
      /* T00 T01 T16 T17 */
      /* T02 T03 T18 T19 */
      /* T04 T05 T20 T21 */
      /* T06 T07 T22 T23 */
      /* T08 T09 T24 T25 */
      /* T10 T11 T26 T27 */
      /* T12 T13 T28 T29 */
      /* T14 T15 T30 T31 */
      const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 8 / 32 % BLOCK_N;
      const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 8 / 32 / BLOCK_N * 32;
      const int n_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int n_global      = n_block_offset + n_loop_offset + n_lane_offset;
      const int k_global      = k_block_offset + k_loop_offset + k_lane_offset;
      FETCH_FLOAT2(B_ldg_reg[n_loop], B_transposed[OFFSET(n_global, k_global, K)]);
    }
    for (int m_loop = 0; m_loop < M_LDG_PER_THREAD; ++m_loop) {
      const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 % BLOCK_M;
      const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 / BLOCK_M * 32;
      const int m_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int m_sm          = m_loop_offset + m_lane_offset;
      const int k_sm          = k_loop_offset + k_lane_offset;
      STORE_FLOAT2(A_sm[k_sm / 16][m_sm][k_sm % 16], A_ldg_reg[m_loop]);
    }
    for (int n_loop = 0; n_loop < N_LDG_PER_THREAD; ++n_loop) {
      const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 % BLOCK_N;
      const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 / BLOCK_N * 64;
      const int n_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int n_sm          = n_loop_offset + n_lane_offset;
      const int k_sm          = k_loop_offset + k_lane_offset;
      STORE_FLOAT2(B_sm[k_sm / 16][n_sm][k_sm % 16], B_ldg_reg[n_loop]);
    }
    FETCH_FLOAT(B_scale_reg, B_scale_transposed[OFFSET(n_block_offset / 128, k_block_offset / 128, K / 128)]);
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      FETCH_FLOAT2(
        A_scale_reg[mg],
        A_scale_transposed[OFFSET(k_block_offset / 128, m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2, M)]);
    }
    __syncthreads();
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      A_scale_reg[mg][0] *= B_scale_reg;
      A_scale_reg[mg][1] *= B_scale_reg;
    }
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      /* 8 x 32 per group, ldmatrix.m8n8.x2.b16 */
      int m_group_offset = m_warp_offset + mg * 8;
      ldmatrix_sync_aligned_m8n8_x2_b16(A_cal_reg[mg].ldm, &A_sm[lane_id / 8][m_group_offset + lane_id % 8][0]);
    }
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      /* 16 x 32 per group, ldmatrix.m8n8.x4.b16 */
      int n_group_offset = n_warp_offset + ng * 16;
      ldmatrix_sync_aligned_m8n8_x4_b16(B_cal_reg[ng].ldm, &B_sm[lane_id / 16][n_group_offset + lane_id % 16][0]);
    }
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
        float C_partial_mma_reg[4] = {0};
        mma_m16n8k32_row_col(C_partial_mma_reg, B_cal_reg[ng].mma, A_cal_reg[mg].mma, C_partial_mma_reg);
        C_cal_reg[mg][ng][0] += C_partial_mma_reg[0] * A_scale_reg[mg][0];
        C_cal_reg[mg][ng][1] += C_partial_mma_reg[1] * A_scale_reg[mg][1];
        C_cal_reg[mg][ng][2] += C_partial_mma_reg[2] * A_scale_reg[mg][0];
        C_cal_reg[mg][ng][3] += C_partial_mma_reg[3] * A_scale_reg[mg][1];
      }
    }
    __syncthreads();
  }
  constexpr int m_lane_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  constexpr int n_lane_offset[4] = {0, 8, 4, 12};
  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      LLMMM::shfl_1_and_0(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_3_and_2(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_23_and_01(C_cal_reg[mg][ng], 0x8, lane_id);
      const int m_global = m_block_offset + m_warp_offset + mg * 8 + m_lane_offset[lane_id % 8];
      const int n_global = n_block_offset + n_warp_offset + ng * 16 + n_lane_offset[lane_id / 8];
      STORE_FLOAT4(C[OFFSET(m_global, n_global, N)], C_cal_reg[mg][ng]);
    }
  }
}

void fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32__partial_loop_k(
  const __nv_fp8_e4m3* A,
  const float*         A_scale_transposed,
  const __nv_fp8_e4m3* B_transposed,
  const float*         B_scale_transposed,
  float*               C,
  int                  M,
  int                  N,
  int                  K,
  cudaStream_t         stream)
{
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 128;
  constexpr int LOOP_K  = 128;
  constexpr int WARP_M  = 64;
  constexpr int WARP_N  = 64;
  static_assert(BLOCK_M > 0 && BLOCK_M <= 128 && BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N == 128 && BLOCK_N % WARP_N == 0);
  static_assert(WARP_M > 0 && WARP_M % 8 == 0);  // mma.m16n8k32.row.col, A is n8k32, B is m16k32
  static_assert(WARP_N > 0 && WARP_N % 16 == 0);
  static_assert(LOOP_K == 128);
  constexpr int WARP_COUNT = BLOCK_M / WARP_M * BLOCK_N / WARP_N;
  static_assert(0 < WARP_COUNT && WARP_COUNT <= 4 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);
  if (!(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % LOOP_K == 0)) {
    throw std::runtime_error("M or N or K are not aligned.");
  }
  dim3 grid(N / BLOCK_N, M / BLOCK_M);
  dim3 block(WARP_COUNT * 32);
  fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32__partial_loop_k<BLOCK_M,
                                                                                    BLOCK_N,
                                                                                    WARP_M,
                                                                                    WARP_N,
                                                                                    LOOP_K>
    <<<grid, block, 0, stream>>>(A, A_scale_transposed, B_transposed, B_scale_transposed, C, M, N, K);
}

template<int BLOCK_M, int BLOCK_N, int QUANT_M, int QUANT_N, bool SCALE_TRANPOSE>
__global__ void fp8_blockwise_symmetric_quantization(const float* x, __nv_fp8_e4m3* q, float* scale, int M, int N)
{
  const int m_block_offset = BLOCK_M * blockIdx.y;
  const int n_block_offset = BLOCK_N * blockIdx.x;
  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;

  constexpr int THREAD_COUNT = 128;
  constexpr int WARP_COUNT   = THREAD_COUNT / 32;

  constexpr int DATA_PER_THREAD = BLOCK_M * BLOCK_N / THREAD_COUNT;
  static_assert(DATA_PER_THREAD == 128);
  constexpr int LOOP_COUNT = DATA_PER_THREAD / 4;

  const float fp8_e4m3_range = 448;

  const float max_float = INT_MIN;

  float max = max_float;

  const int scale_N = (N + QUANT_N - 1) / QUANT_N;

  __shared__ float x_sm[QUANT_M * QUANT_N];

  for (int loop = 0; loop < LOOP_COUNT; ++loop) {
    int m = m_block_offset + loop * WARP_COUNT + warp_id;
    if (m < M) {
      float data[4];
      FETCH_FLOAT4_PREFETCH_256B_WITH_SRC_PTR(data, &x[OFFSET(m, n_block_offset + lane_id * 4, N)]);
      if (QUANT_M == 128) {
        STORE_FLOAT4(x_sm[OFFSET(loop * WARP_COUNT + warp_id, lane_id * 4, BLOCK_N)], data);
      }
      max = (max < fabs(data[0])) ? fabs(data[0]) : max;
      max = (max < fabs(data[1])) ? fabs(data[1]) : max;
      max = (max < fabs(data[2])) ? fabs(data[2]) : max;
      max = (max < fabs(data[3])) ? fabs(data[3]) : max;
      if constexpr (QUANT_M == 1) {
        max     = warp_reduce_max(max);
        max     = warp_broadcast(0, max);
        float s = max / fp8_e4m3_range;

        __nv_fp8_e4m3 quanted[4] = {
          __nv_fp8_e4m3(s ? data[0] / s : 0.0f),
          __nv_fp8_e4m3(s ? data[1] / s : 0.0f),
          __nv_fp8_e4m3(s ? data[2] / s : 0.0f),
          __nv_fp8_e4m3(s ? data[3] / s : 0.0f),
        };

        max = max_float;

        STORE_FLOAT(q[OFFSET(m, n_block_offset + lane_id * 4, N)], quanted);
        static_assert(BLOCK_N == QUANT_N);
        if (lane_id == 0) {
          if constexpr (SCALE_TRANPOSE) {
            STORE_FLOAT(scale[OFFSET(n_block_offset / QUANT_N, m, M)], s);
          }
          else {
            STORE_FLOAT(scale[OFFSET(m, n_block_offset / QUANT_N, scale_N)], s);
          }
        }
      }
    }
  }
  if constexpr (QUANT_M == 128) {
    __shared__ float block_max_value[WARP_COUNT];
    max = warp_reduce_max(max);
    if (lane_id == 0) {
      static_assert(WARP_COUNT == 4);
      block_max_value[warp_id] = max;
    }
    __syncthreads();
    float max4[4];
    FETCH_FLOAT4(max4[0], block_max_value[0]);
    max     = (max4[0] > max4[1]) ? max4[0] : max4[1];
    max     = (max > max4[2]) ? max : max4[2];
    max     = (max > max4[3]) ? max : max4[3];
    float s = max / fp8_e4m3_range;
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
      int m = m_block_offset + loop * WARP_COUNT + warp_id;
      if (m < M) {
        float data[4];
        FETCH_FLOAT4(data[0], x_sm[OFFSET(loop * WARP_COUNT + warp_id, lane_id * 4, BLOCK_N)]);

        __nv_fp8_e4m3 quanted[4] = {
          __nv_fp8_e4m3(s ? data[0] / s : 0.0f),
          __nv_fp8_e4m3(s ? data[1] / s : 0.0f),
          __nv_fp8_e4m3(s ? data[2] / s : 0.0f),
          __nv_fp8_e4m3(s ? data[3] / s : 0.0f),
        };
        STORE_FLOAT(q[OFFSET(m_block_offset + loop * WARP_COUNT + warp_id, n_block_offset + lane_id * 4, N)], quanted);
      }
      static_assert(BLOCK_N == QUANT_N);
      if (threadIdx.x == 0) {
        if constexpr (SCALE_TRANPOSE) {
          STORE_FLOAT(
            scale[OFFSET(n_block_offset / QUANT_N, (m_block_offset + loop * WARP_COUNT + warp_id) / QUANT_M, M)], s);
        }
        else {
          STORE_FLOAT(
            scale[OFFSET((m_block_offset + loop * WARP_COUNT + warp_id) / QUANT_M, n_block_offset / QUANT_N, scale_N)],
            s);
        }
      }
    }
  }
}

template<int QUANT_M, int QUANT_N, bool SCALE_TRANPOSE>
void fp8_blockwise_symmetric_quantization(
  const float* x, __nv_fp8_e4m3* q, float* scale, int M, int N, cudaStream_t stream)
{
  static_assert(QUANT_M == 1 || QUANT_M == 128);
  static_assert(QUANT_N == 128);
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_N = 128;

  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  dim3 block(128);
  auto kSmemSize   = 0;
  auto kernel_func = &fp8_blockwise_symmetric_quantization<BLOCK_M, BLOCK_N, QUANT_M, QUANT_N, SCALE_TRANPOSE>;
  CHECK_CUDA_RETURN(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
  fp8_blockwise_symmetric_quantization<BLOCK_M, BLOCK_N, QUANT_M, QUANT_N, SCALE_TRANPOSE>
    <<<grid, block, 0, stream>>>(x, q, scale, M, N);
}

template<int BLOCK_M, int BLOCK_N, int WARP_M, int WARP_N, int LOOP_K>
__global__ void fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8(const __nv_fp8_e4m3* A,
                                                                                 const float* A_scale_transposed,
                                                                                 const __nv_fp8_e4m3* B_transposed,
                                                                                 const float*   B_scale_transposed,
                                                                                 __nv_fp8_e4m3* C,
                                                                                 float*         C_scale_transposed,
                                                                                 int            M,
                                                                                 int            N,
                                                                                 int            K)
{
  constexpr int M_WARP_COUNT     = BLOCK_M / WARP_M;
  constexpr int N_WARP_COUNT     = BLOCK_N / WARP_N;
  constexpr int WARP_COUNT       = M_WARP_COUNT * N_WARP_COUNT;
  constexpr int THREAD_COUNT     = WARP_COUNT * 32;
  constexpr int M_GROUP_PER_WARP = WARP_M / 8;
  constexpr int N_GROUP_PER_WARP = WARP_N / 16;

  using fp8_t = __nv_fp8_e4m3;

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = BLOCK_M * blockIdx.y;
  const int n_block_offset = BLOCK_N * blockIdx.x;
  const int m_warp_id      = warp_id % M_WARP_COUNT;
  const int n_warp_id      = warp_id / M_WARP_COUNT;
  const int m_warp_offset  = m_warp_id * WARP_M;
  const int n_warp_offset  = n_warp_id * WARP_N;

  // LDG
  constexpr int PARTIAL_LOOP_K = 32;
  static_assert(PARTIAL_LOOP_K == 32 && LOOP_K == PARTIAL_LOOP_K * 4);
  __shared__ fp8_t A_sm[PARTIAL_LOOP_K / 16][BLOCK_M][16];
  float            A_scale_reg[M_GROUP_PER_WARP][2];  // A_scale_transposed is (K/128) x M
  __shared__ fp8_t B_sm[PARTIAL_LOOP_K / 16][BLOCK_N][16];
  float            B_scale_reg;  // B_scale_transposed is (N/128) x (K/128)
  __shared__ float C_block_extrema[M_WARP_COUNT][WARP_M][N_WARP_COUNT];

  // PARTIAL_LOOP_K is aimed at reducing register usage.
  constexpr int BYTE_PER_LDG      = sizeof(float2);
  constexpr int M_BYTE_PER_THREAD = BLOCK_M * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(M_BYTE_PER_THREAD % BYTE_PER_LDG == 0);
  constexpr int N_BYTE_PER_THREAD = BLOCK_N * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(N_BYTE_PER_THREAD % BYTE_PER_LDG == 0);

  constexpr int M_LDG_PER_THREAD = M_BYTE_PER_THREAD / BYTE_PER_LDG;
  constexpr int N_LDG_PER_THREAD = N_BYTE_PER_THREAD / BYTE_PER_LDG;

  fp8_t A_ldg_reg[M_LDG_PER_THREAD][BYTE_PER_LDG];
  fp8_t B_ldg_reg[N_LDG_PER_THREAD][BYTE_PER_LDG];

  // MMA
  union {
    uint16_t ldm[2][2];
    fp8_t    mma[8];
  } A_cal_reg[M_GROUP_PER_WARP];
  union {
    uint16_t ldm[4][2];
    fp8_t    mma[16];
  } B_cal_reg[N_GROUP_PER_WARP];

  float C_cal_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4] = {0};

  for (int k_block_offset = 0; k_block_offset < K; k_block_offset += PARTIAL_LOOP_K) {
    for (int m_loop = 0; m_loop < M_LDG_PER_THREAD; ++m_loop) {
      /* T00 T01 T16 T17 */
      /* T02 T03 T18 T19 */
      /* T04 T05 T20 T21 */
      /* T06 T07 T22 T23 */
      /* T08 T09 T24 T25 */
      /* T10 T11 T26 T27 */
      /* T12 T13 T28 T29 */
      /* T14 T15 T30 T31 */
      const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 % BLOCK_M;
      const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 / BLOCK_M * 32;
      const int m_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int m_global      = m_block_offset + m_loop_offset + m_lane_offset;
      const int k_global      = k_block_offset + k_loop_offset + k_lane_offset;
      FETCH_FLOAT2(A_ldg_reg[m_loop], A[OFFSET(m_global, k_global, K)]);
    }
    for (int n_loop = 0; n_loop < N_LDG_PER_THREAD; ++n_loop) {
      /* T00 T01 T16 T17 */
      /* T02 T03 T18 T19 */
      /* T04 T05 T20 T21 */
      /* T06 T07 T22 T23 */
      /* T08 T09 T24 T25 */
      /* T10 T11 T26 T27 */
      /* T12 T13 T28 T29 */
      /* T14 T15 T30 T31 */
      const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 8 / 32 % BLOCK_N;
      const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 8 / 32 / BLOCK_N * 32;
      const int n_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int n_global      = n_block_offset + n_loop_offset + n_lane_offset;
      const int k_global      = k_block_offset + k_loop_offset + k_lane_offset;
      FETCH_FLOAT2(B_ldg_reg[n_loop], B_transposed[OFFSET(n_global, k_global, K)]);
    }
    for (int m_loop = 0; m_loop < M_LDG_PER_THREAD; ++m_loop) {
      const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 % BLOCK_M;
      const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 / BLOCK_M * 32;
      const int m_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int m_sm          = m_loop_offset + m_lane_offset;
      const int k_sm          = k_loop_offset + k_lane_offset;
      STORE_FLOAT2(A_sm[k_sm / 16][m_sm][k_sm % 16], A_ldg_reg[m_loop]);
    }
    for (int n_loop = 0; n_loop < N_LDG_PER_THREAD; ++n_loop) {
      const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 % BLOCK_N;
      const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 / BLOCK_N * 64;
      const int n_lane_offset = lane_id % 16 / 2;
      const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);
      const int n_sm          = n_loop_offset + n_lane_offset;
      const int k_sm          = k_loop_offset + k_lane_offset;
      STORE_FLOAT2(B_sm[k_sm / 16][n_sm][k_sm % 16], B_ldg_reg[n_loop]);
    }
    FETCH_FLOAT(B_scale_reg, B_scale_transposed[OFFSET(n_block_offset / 128, k_block_offset / 128, K / 128)]);
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      FETCH_FLOAT2(
        A_scale_reg[mg],
        A_scale_transposed[OFFSET(k_block_offset / 128, m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2, M)]);
    }
    __syncthreads();
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      A_scale_reg[mg][0] *= B_scale_reg;
      A_scale_reg[mg][1] *= B_scale_reg;
    }
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      /* 8 x 32 per group, ldmatrix.m8n8.x2.b16 */
      int m_group_offset = m_warp_offset + mg * 8;
      ldmatrix_sync_aligned_m8n8_x2_b16(A_cal_reg[mg].ldm, &A_sm[lane_id / 8][m_group_offset + lane_id % 8][0]);
    }
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      /* 16 x 32 per group, ldmatrix.m8n8.x4.b16 */
      int n_group_offset = n_warp_offset + ng * 16;
      ldmatrix_sync_aligned_m8n8_x4_b16(B_cal_reg[ng].ldm, &B_sm[lane_id / 16][n_group_offset + lane_id % 16][0]);
    }
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
        float C_partial_mma_reg[4] = {0};
        mma_m16n8k32_row_col(C_partial_mma_reg, B_cal_reg[ng].mma, A_cal_reg[mg].mma, C_partial_mma_reg);
        C_cal_reg[mg][ng][0] += C_partial_mma_reg[0] * A_scale_reg[mg][0];
        C_cal_reg[mg][ng][1] += C_partial_mma_reg[1] * A_scale_reg[mg][1];
        C_cal_reg[mg][ng][2] += C_partial_mma_reg[2] * A_scale_reg[mg][0];
        C_cal_reg[mg][ng][3] += C_partial_mma_reg[3] * A_scale_reg[mg][1];
      }
    }
    __syncthreads();
  }

  constexpr int m_lane_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  constexpr int n_lane_offset[4] = {0, 8, 4, 12};
  float max_val[M_GROUP_PER_WARP];
  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    max_val[mg] = fabs(C_cal_reg[mg][0][0]);
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      LLMMM::shfl_1_and_0(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_3_and_2(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_23_and_01(C_cal_reg[mg][ng], 0x8, lane_id);

      /* after shfl, m ↓ n →, m1n4 per thread */
      /* T00 T16 T08 T24 */
      /* T04 T20 T12 T28 */
      /* T01 T17 T09 T25 */
      /* T05 T21 T13 T29 */
      /* T02 T18 T10 T26 */
      /* T06 T22 T14 T30 */
      /* T03 T19 T11 T27 */
      /* T07 T23 T15 T31 */

      constexpr int array_size = get_array_size(C_cal_reg[0][0]);
      for (int i = 0; i < array_size; ++i) {
        max_val[mg] = max_val[mg] > fabs(C_cal_reg[mg][ng][i]) ? max_val[mg] : fabs(C_cal_reg[mg][ng][i]);
      }
      max_val[mg] = max(max_val[mg], __shfl_xor_sync(0xffffffff, max_val[mg], 0x10));
      max_val[mg] = max(max_val[mg], __shfl_xor_sync(0xffffffff, max_val[mg], 0x08));
    }
    if (lane_id < 8) {
      const int m = mg * 8 + m_lane_offset[lane_id];
      STORE_FLOAT(C_block_extrema[m_warp_id][m][n_warp_id], max_val[mg]);
    }
  }

  __syncthreads();

  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    const int m = mg * 8 + m_lane_offset[lane_id % 8];
    for (int nw = 0; nw < N_WARP_COUNT; ++nw) {
      float max_val_sm;
      FETCH_FLOAT(max_val_sm, C_block_extrema[m_warp_id][m][nw]);
      max_val[mg] = max(max_val[mg], max_val_sm);
    }
  }

  constexpr float fp8_e4m3_range = 448;

  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    float     scale    = max_val[mg] / fp8_e4m3_range;
    const int m_global = m_block_offset + m_warp_offset + mg * 8 + m_lane_offset[lane_id % 8];
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      const int n_global = n_block_offset + n_warp_offset + ng * 16 + n_lane_offset[lane_id / 8];
      fp8_t     q[4]     = {fp8_t(C_cal_reg[mg][ng][0] / scale),
                            fp8_t(C_cal_reg[mg][ng][1] / scale),
                            fp8_t(C_cal_reg[mg][ng][2] / scale),
                            fp8_t(C_cal_reg[mg][ng][3] / scale)};
      STORE_FLOAT(C[OFFSET(m_global, n_global, N)], q);
    }
    if (lane_id < 8) {
      static_assert(BLOCK_N <= 128);
      STORE_FLOAT(C_scale_transposed[OFFSET(n_block_offset / 128, m_global, M)], scale);
    }
  }
}

template<int BLOCK_M, int BLOCK_N, int WARP_M, int WARP_N, int LOOP_K>
__global__ void fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__opt_reg(const __nv_fp8_e4m3* A,
                                                                                 const float* A_scale_transposed,
                                                                                 const __nv_fp8_e4m3* B_transposed,
                                                                                 const float*   B_scale_transposed,
                                                                                 __nv_fp8_e4m3* C,
                                                                                 float*         C_scale_transposed,
                                                                                 int            M,
                                                                                 int            N,
                                                                                 int            K)
{
  constexpr int M_WARP_COUNT     = BLOCK_M / WARP_M;
  constexpr int N_WARP_COUNT     = BLOCK_N / WARP_N;
  constexpr int WARP_COUNT       = M_WARP_COUNT * N_WARP_COUNT;
  constexpr int THREAD_COUNT     = WARP_COUNT * 32;
  constexpr int M_GROUP_PER_WARP = WARP_M / 8;
  constexpr int N_GROUP_PER_WARP = WARP_N / 16;

  using fp8_t = __nv_fp8_e4m3;

  const int warp_id        = threadIdx.x / 32;
  const int lane_id        = threadIdx.x % 32;
  const int m_block_offset = BLOCK_M * blockIdx.y;
  const int n_block_offset = BLOCK_N * blockIdx.x;
  const int m_warp_id      = warp_id % M_WARP_COUNT;
  const int n_warp_id      = warp_id / M_WARP_COUNT;
  const int m_warp_offset  = m_warp_id * WARP_M;
  const int n_warp_offset  = n_warp_id * WARP_N;

  // LDG
  constexpr int PARTIAL_LOOP_K = 32;
  static_assert(PARTIAL_LOOP_K == 32 && LOOP_K == PARTIAL_LOOP_K * 4);
  __shared__ fp8_t A_sm[PARTIAL_LOOP_K / 16][BLOCK_M][16];
  float            A_scale_reg[M_GROUP_PER_WARP][2];  // A_scale_transposed is (K/128) x M
  __shared__ fp8_t B_sm[PARTIAL_LOOP_K / 16][BLOCK_N][16];
  float            B_scale_reg;  // B_scale_transposed is (N/128) x (K/128)
  __shared__ float C_block_extrema[M_WARP_COUNT][WARP_M][N_WARP_COUNT];

  // PARTIAL_LOOP_K is aimed at reducing register usage.
  constexpr int BYTE_PER_LDG      = sizeof(float2);
  constexpr int M_BYTE_PER_THREAD = BLOCK_M * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(M_BYTE_PER_THREAD % BYTE_PER_LDG == 0);
  constexpr int N_BYTE_PER_THREAD = BLOCK_N * PARTIAL_LOOP_K / THREAD_COUNT;
  static_assert(N_BYTE_PER_THREAD % BYTE_PER_LDG == 0);

  constexpr int M_LDG_PER_THREAD = M_BYTE_PER_THREAD / BYTE_PER_LDG;
  constexpr int N_LDG_PER_THREAD = N_BYTE_PER_THREAD / BYTE_PER_LDG;

  fp8_t A_ldg_reg[M_LDG_PER_THREAD][BYTE_PER_LDG];
  fp8_t B_ldg_reg[N_LDG_PER_THREAD][BYTE_PER_LDG];

  // MMA
  union {
    uint16_t ldm[2][2];
    fp8_t    mma[8];
  } A_cal_reg[M_GROUP_PER_WARP];
  union {
    uint16_t ldm[4][2];
    fp8_t    mma[16];
  } B_cal_reg[N_GROUP_PER_WARP];

  constexpr int C_PER_THREAD = WARP_M * WARP_N / 32;
  static_assert(C_PER_THREAD < 128);
  float           C_mma_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4];
  float           C_cal_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4] = {0};
  constexpr float ZERO_ARR[4]                                      = {0, 0, 0, 0};

  const int A_ldg_m_offset         = m_block_offset + warp_id * 32 * 8 / 32 + lane_id % 16 / 2;
  const int B_ldg_n_offset         = n_block_offset + warp_id * 32 * 8 / 32 + lane_id % 16 / 2;
  const int A_ldg_and_sts_k_offset = (lane_id & 1) * 8 + (lane_id & 16);
  const int B_ldg_and_sts_k_offset = (lane_id & 1) * 8 + (lane_id & 16);
  const int A_sts_m_offset         = warp_id * 32 * 8 / 32 + lane_id % 16 / 2;
  const int B_sts_n_offset         = warp_id * 32 * 8 / 32 + lane_id % 16 / 2;

  constexpr int m_lane_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  constexpr int n_lane_offset[4] = {0, 8, 4, 12};
  float         max_val[M_GROUP_PER_WARP];

  enum {
    CAL_OFF = 0,
    CAL_ON_FIRST,
    CAL_ON_MIDDLE,
    CAL_ON_LAST,
  };

#define ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, rank)                               \
  {                                                                                                                    \
    if constexpr (cal_switch && rank < MxN_GROUP_PER_WARP) {                                                           \
      constexpr int mg = rank % M_GROUP_PER_WARP;                                                                      \
      constexpr int ng = rank / M_GROUP_PER_WARP;                                                                      \
      if constexpr (cal_switch == CAL_ON_FIRST) {                                                                      \
        mma_m16n8k32_row_col(C_mma_reg[mg][ng], B_cal_reg[ng].mma, A_cal_reg[mg].mma, ZERO_ARR);                       \
      }                                                                                                                \
      else {                                                                                                           \
        mma_m16n8k32_row_col(C_mma_reg[mg][ng], B_cal_reg[ng].mma, A_cal_reg[mg].mma, C_mma_reg[mg][ng]);              \
      }                                                                                                                \
      if constexpr (cal_switch == CAL_ON_LAST) {                                                                       \
        C_cal_reg[mg][ng][0] += C_mma_reg[mg][ng][0] * A_scale_reg[mg][0];                                             \
        C_cal_reg[mg][ng][1] += C_mma_reg[mg][ng][1] * A_scale_reg[mg][1];                                             \
        C_cal_reg[mg][ng][2] += C_mma_reg[mg][ng][2] * A_scale_reg[mg][0];                                             \
        C_cal_reg[mg][ng][3] += C_mma_reg[mg][ng][3] * A_scale_reg[mg][1];                                             \
      }                                                                                                                \
    }                                                                                                                  \
    if constexpr (ldg_switch && rank < M_LDG_PER_THREAD) {                                                             \
      constexpr int m_loop        = rank;                                                                              \
      constexpr int m_loop_offset = m_loop * THREAD_COUNT * 8 / 32;                                                    \
      /* T00 T01 T16 T17 */                                                                                            \
      /* T02 T03 T18 T19 */                                                                                            \
      /* T04 T05 T20 T21 */                                                                                            \
      /* T06 T07 T22 T23 */                                                                                            \
      /* T08 T09 T24 T25 */                                                                                            \
      /* T10 T11 T26 T27 */                                                                                            \
      /* T12 T13 T28 T29 */                                                                                            \
      /* T14 T15 T30 T31 */                                                                                            \
      /* const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 % BLOCK_M;        */                \
      /* const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 / BLOCK_M * 32;   */                \
      /* const int m_lane_offset = lane_id % 16 / 2;                                                 */                \
      /* const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);                               */                \
      /* const int m_global      = A_ldg_m_offset + m_loop * THREAD_COUNT * / 32;                    */                \
      /* const int k_global      = k_block_offset + k_loop_offset + k_lane_offset;                   */                \
      const int m_global = A_ldg_m_offset + m_loop_offset;                                                             \
      const int k_global = A_ldg_and_sts_k_offset + k_block_offset;                                                    \
      FETCH_FLOAT2(A_ldg_reg[m_loop], A[OFFSET(m_global, k_global, K)]);                                               \
    }                                                                                                                  \
    if constexpr (ldg_switch && rank < N_LDG_PER_THREAD) {                                                             \
      constexpr int n_loop        = rank;                                                                              \
      constexpr int n_loop_offset = n_loop * THREAD_COUNT * 8 / 32;                                                    \
      /* T00 T01 T16 T17 */                                                                                            \
      /* T02 T03 T18 T19 */                                                                                            \
      /* T04 T05 T20 T21 */                                                                                            \
      /* T06 T07 T22 T23 */                                                                                            \
      /* T08 T09 T24 T25 */                                                                                            \
      /* T10 T11 T26 T27 */                                                                                            \
      /* T12 T13 T28 T29 */                                                                                            \
      /* T14 T15 T30 T31 */                                                                                            \
      /* const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 8 / 32 % BLOCK_N;       */                 \
      /* const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 8 / 32 / BLOCK_N * 32;  */                 \
      /* const int n_lane_offset = lane_id % 16 / 2;                                                */                 \
      /* const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);                              */                 \
      /* const int n_global      = n_block_offset + n_loop_offset + n_lane_offset;                  */                 \
      /* const int k_global      = k_block_offset + k_loop_offset + k_lane_offset;                  */                 \
      const int n_global = B_ldg_n_offset + n_loop_offset;                                                             \
      const int k_global = B_ldg_and_sts_k_offset + k_block_offset;                                                    \
      FETCH_FLOAT2(B_ldg_reg[n_loop], B_transposed[OFFSET(n_global, k_global, K)]);                                    \
    }                                                                                                                  \
    if constexpr (sts_switch && rank < M_LDG_PER_THREAD) {                                                             \
      constexpr int m_loop        = rank;                                                                              \
      constexpr int m_loop_offset = m_loop * THREAD_COUNT * 8 / 32;                                                    \
      /* const int m_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 % BLOCK_M;      */                  \
      /* const int k_loop_offset = (warp_id * 32 + m_loop * THREAD_COUNT) * 8 / 32 / BLOCK_M * 32; */                  \
      /* const int m_lane_offset = lane_id % 16 / 2;                                               */                  \
      /* const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);                             */                  \
      /* const int m_sm          = m_loop_offset + m_lane_offset;                                  */                  \
      /* const int k_sm          = k_loop_offset + k_lane_offset;                                  */                  \
      const int m_sm = m_loop_offset + A_sts_m_offset;                                                                 \
      const int k_sm = A_ldg_and_sts_k_offset;                                                                         \
      STORE_FLOAT2(A_sm[k_sm / 16][m_sm][k_sm % 16], A_ldg_reg[m_loop]);                                               \
    }                                                                                                                  \
    if constexpr (sts_switch && rank < N_LDG_PER_THREAD) {                                                             \
      constexpr int n_loop        = rank;                                                                              \
      constexpr int n_loop_offset = n_loop * THREAD_COUNT * 8 / 32;                                                    \
      /* const int n_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 % BLOCK_N;      */                 \
      /* const int k_loop_offset = (warp_id * 32 + n_loop * THREAD_COUNT) * 16 / 64 / BLOCK_N * 64; */                 \
      /* const int n_lane_offset = lane_id % 16 / 2;                                                */                 \
      /* const int k_lane_offset = (lane_id & 1) * 8 + (lane_id & 16);                              */                 \
      /* const int n_sm          = n_loop_offset + n_lane_offset;                                   */                 \
      /* const int k_sm          = k_loop_offset + k_lane_offset;                                   */                 \
      const int n_sm = n_loop_offset + B_sts_n_offset;                                                                 \
      const int k_sm = B_ldg_and_sts_k_offset;                                                                         \
      STORE_FLOAT2(B_sm[k_sm / 16][n_sm][k_sm % 16], B_ldg_reg[n_loop]);                                               \
    }                                                                                                                  \
    if constexpr (rd_switch && rank < MxN_GROUP_PER_WARP) {                                                            \
      constexpr int mg = rank % M_GROUP_PER_WARP;                                                                      \
      constexpr int ng = rank / M_GROUP_PER_WARP;                                                                      \
      if constexpr (mg == 0 && ng == 0) {                                                                              \
        max_val[mg] = fabs(C_cal_reg[mg][0][0]);                                                                       \
      }                                                                                                                \
      LLMMM::shfl_1_and_0(C_cal_reg[mg][ng], 0x4, lane_id);                                                            \
      LLMMM::shfl_3_and_2(C_cal_reg[mg][ng], 0x4, lane_id);                                                            \
      LLMMM::shfl_23_and_01(C_cal_reg[mg][ng], 0x8, lane_id);                                                          \
      /* after shfl, m ↓ n →, m1n4 per thread */                                                                   \
      /* T00 T16 T08 T24                      */                                                                       \
      /* T04 T20 T12 T28                      */                                                                       \
      /* T01 T17 T09 T25                      */                                                                       \
      /* T05 T21 T13 T29                      */                                                                       \
      /* T02 T18 T10 T26                      */                                                                       \
      /* T06 T22 T14 T30                      */                                                                       \
      /* T03 T19 T11 T27                      */                                                                       \
      /* T07 T23 T15 T31                      */                                                                       \
                                                                                                                       \
      constexpr int array_size = get_array_size(C_cal_reg[0][0]);                                                      \
      for (int i = 0; i < array_size; ++i) {                                                                           \
        max_val[mg] = max_val[mg] > fabs(C_cal_reg[mg][ng][i]) ? max_val[mg] : fabs(C_cal_reg[mg][ng][i]);             \
      }                                                                                                                \
      max_val[mg] = max(max_val[mg], __shfl_xor_sync(0xffffffff, max_val[mg], 0x10));                                  \
      max_val[mg] = max(max_val[mg], __shfl_xor_sync(0xffffffff, max_val[mg], 0x08));                                  \
      if constexpr (ng == N_GROUP_PER_WARP - 1) {                                                                      \
        if (lane_id < 8) {                                                                                             \
          const int m = mg * 8 + m_lane_offset[lane_id];                                                               \
          STORE_FLOAT(C_block_extrema[m_warp_id][m][n_warp_id], max_val[mg]);                                          \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

#define ldg__sts__cal__rd(ldg_switch, sts_switch, cal_switch, rd_switch)                                               \
  {                                                                                                                    \
    constexpr int MxN_GROUP_PER_WARP = M_GROUP_PER_WARP * N_GROUP_PER_WARP;                                            \
    static_assert(MxN_GROUP_PER_WARP <= 32);                                                                           \
    static_assert(M_LDG_PER_THREAD <= MxN_GROUP_PER_WARP);                                                             \
    static_assert(N_LDG_PER_THREAD <= MxN_GROUP_PER_WARP);                                                             \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 0);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 1);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 2);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 3);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 4);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 5);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 6);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 7);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 8);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 9);                                     \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 10);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 11);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 12);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 13);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 14);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 15);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 16);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 17);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 18);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 19);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 20);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 21);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 22);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 23);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 24);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 25);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 26);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 27);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 28);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 29);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 30);                                    \
    ldg__sts__cal__rd__per_rank(ldg_switch, sts_switch, cal_switch, rd_switch, 31);                                    \
  }

  int k_block_offset = 0;
  while (k_block_offset < K) {
    FETCH_FLOAT(B_scale_reg, B_scale_transposed[OFFSET(n_block_offset / 128, k_block_offset / 128, K / 128)]);
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      FETCH_FLOAT2(
        A_scale_reg[mg],
        A_scale_transposed[OFFSET(k_block_offset / 128, m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2, M)]);
    }
    for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
      A_scale_reg[mg][0] *= B_scale_reg;
      A_scale_reg[mg][1] *= B_scale_reg;
    }
    {
      ldg__sts__cal__rd(true, false, CAL_OFF, false);
      ldg__sts__cal__rd(false, true, CAL_OFF, false);
      __syncthreads();
      for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
        /* 8 x 32 per group, ldmatrix.m8n8.x2.b16 */
        int m_group_offset = m_warp_offset + mg * 8;
        ldmatrix_sync_aligned_m8n8_x2_b16(A_cal_reg[mg].ldm, &A_sm[lane_id / 8][m_group_offset + lane_id % 8][0]);
      }
      for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
        /* 16 x 32 per group, ldmatrix.m8n8.x4.b16 */
        int n_group_offset = n_warp_offset + ng * 16;
        ldmatrix_sync_aligned_m8n8_x4_b16(B_cal_reg[ng].ldm, &B_sm[lane_id / 16][n_group_offset + lane_id % 16][0]);
      }
      __syncthreads();
      ldg__sts__cal__rd(false, false, CAL_ON_FIRST, false);
      k_block_offset += PARTIAL_LOOP_K;
    }
    {
      ldg__sts__cal__rd(true, false, CAL_OFF, false);
      ldg__sts__cal__rd(false, true, CAL_OFF, false);
      __syncthreads();
      for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
        /* 8 x 32 per group, ldmatrix.m8n8.x2.b16 */
        int m_group_offset = m_warp_offset + mg * 8;
        ldmatrix_sync_aligned_m8n8_x2_b16(A_cal_reg[mg].ldm, &A_sm[lane_id / 8][m_group_offset + lane_id % 8][0]);
      }
      for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
        /* 16 x 32 per group, ldmatrix.m8n8.x4.b16 */
        int n_group_offset = n_warp_offset + ng * 16;
        ldmatrix_sync_aligned_m8n8_x4_b16(B_cal_reg[ng].ldm, &B_sm[lane_id / 16][n_group_offset + lane_id % 16][0]);
      }
      __syncthreads();
      ldg__sts__cal__rd(false, false, CAL_ON_MIDDLE, false);
      k_block_offset += PARTIAL_LOOP_K;
    }
    {
      ldg__sts__cal__rd(true, false, CAL_OFF, false);
      ldg__sts__cal__rd(false, true, CAL_OFF, false);
      __syncthreads();
      for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
        /* 8 x 32 per group, ldmatrix.m8n8.x2.b16 */
        int m_group_offset = m_warp_offset + mg * 8;
        ldmatrix_sync_aligned_m8n8_x2_b16(A_cal_reg[mg].ldm, &A_sm[lane_id / 8][m_group_offset + lane_id % 8][0]);
      }
      for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
        /* 16 x 32 per group, ldmatrix.m8n8.x4.b16 */
        int n_group_offset = n_warp_offset + ng * 16;
        ldmatrix_sync_aligned_m8n8_x4_b16(B_cal_reg[ng].ldm, &B_sm[lane_id / 16][n_group_offset + lane_id % 16][0]);
      }
      __syncthreads();
      ldg__sts__cal__rd(false, false, CAL_ON_MIDDLE, false);
      k_block_offset += PARTIAL_LOOP_K;
    }
    {
      ldg__sts__cal__rd(true, false, CAL_OFF, false);
      ldg__sts__cal__rd(false, true, CAL_OFF, false);
      __syncthreads();
      for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
        /* 8 x 32 per group, ldmatrix.m8n8.x2.b16 */
        int m_group_offset = m_warp_offset + mg * 8;
        ldmatrix_sync_aligned_m8n8_x2_b16(A_cal_reg[mg].ldm, &A_sm[lane_id / 8][m_group_offset + lane_id % 8][0]);
      }
      for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
        /* 16 x 32 per group, ldmatrix.m8n8.x4.b16 */
        int n_group_offset = n_warp_offset + ng * 16;
        ldmatrix_sync_aligned_m8n8_x4_b16(B_cal_reg[ng].ldm, &B_sm[lane_id / 16][n_group_offset + lane_id % 16][0]);
      }
      __syncthreads();
      ldg__sts__cal__rd(false, false, CAL_ON_LAST, false);
      k_block_offset += PARTIAL_LOOP_K;
    }
  }

  ldg__sts__cal__rd(false, false, CAL_OFF, true);

  __syncthreads();

  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    const int m = mg * 8 + m_lane_offset[lane_id % 8];
    for (int nw = 0; nw < N_WARP_COUNT; ++nw) {
      float max_val_sm;
      FETCH_FLOAT(max_val_sm, C_block_extrema[m_warp_id][m][nw]);
      max_val[mg] = max(max_val[mg], max_val_sm);
    }
  }

  constexpr float fp8_e4m3_range = 448;

  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    const float scale_inv = fp8_e4m3_range / max_val[mg];
    const float scale     = max_val[mg] / fp8_e4m3_range;
    const int   m_global  = m_block_offset + m_warp_offset + mg * 8 + m_lane_offset[lane_id % 8];
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      const int n_global = n_block_offset + n_warp_offset + ng * 16 + n_lane_offset[lane_id / 8];
      fp8_t     q[4]     = {fp8_t(C_cal_reg[mg][ng][0] * scale_inv),
                            fp8_t(C_cal_reg[mg][ng][1] * scale_inv),
                            fp8_t(C_cal_reg[mg][ng][2] * scale_inv),
                            fp8_t(C_cal_reg[mg][ng][3] * scale_inv)};
      STORE_FLOAT(C[OFFSET(m_global, n_global, N)], q);
    }
    if (lane_id < 8) {
      static_assert(BLOCK_N <= 128);
      STORE_FLOAT(C_scale_transposed[OFFSET(n_block_offset / 128, m_global, M)], scale);
    }
  }
#undef ldg__sts__cal__rd
#undef ldg__sts__cal__rd__per_rank
}

#define launch(function)                                                                                               \
  void function(const __nv_fp8_e4m3* A,                                                                                \
                const float*         A_scale_transposed,                                                               \
                const __nv_fp8_e4m3* B_transposed,                                                                     \
                const float*         B_scale_transposed,                                                               \
                __nv_fp8_e4m3*       C,                                                                                \
                float*               C_scale_transposed,                                                               \
                int                  M,                                                                                \
                int                  N,                                                                                \
                int                  K,                                                                                \
                cudaStream_t         stream)                                                                           \
  {                                                                                                                    \
    constexpr int BLOCK_M = 128;                                                                                       \
    constexpr int BLOCK_N = 128;                                                                                       \
    constexpr int LOOP_K  = 128;                                                                                       \
    constexpr int WARP_M  = 32;                                                                                        \
    constexpr int WARP_N  = 64;                                                                                        \
    static_assert(BLOCK_M > 0 && BLOCK_M <= 128 && BLOCK_M % WARP_M == 0);                                             \
    static_assert(BLOCK_N == 128 && BLOCK_N % WARP_N == 0);                                                            \
    static_assert(WARP_M > 0 && WARP_M % 8 == 0); /* mma.m16n8k32.row.col, A is n8k32, B is m16k32 */                  \
    static_assert(WARP_N > 0 && WARP_N % 16 == 0);                                                                     \
    static_assert(LOOP_K == 128);                                                                                      \
    constexpr int WARP_COUNT   = BLOCK_M / WARP_M * BLOCK_N / WARP_N;                                                  \
    static_assert(0 < WARP_COUNT && WARP_COUNT <= 8 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);                          \
    if (!(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % LOOP_K == 0)) {                                                  \
      throw std::runtime_error("M or N or K are not aligned.");                                                        \
    }                                                                                                                  \
    dim3 grid(N / BLOCK_N, M / BLOCK_M);                                                                               \
    dim3 block(WARP_COUNT * 32);                                                                                       \
    function<BLOCK_M, BLOCK_N, WARP_M, WARP_N, LOOP_K><<<grid, block, 0, stream>>>(                                    \
      A, A_scale_transposed, B_transposed, B_scale_transposed, C, C_scale_transposed, M, N, K);                        \
  }

launch(fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8);
launch(fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__opt_reg);
#undef launch

int main()
{
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);

  std::vector<float>                    h_A(M * K), h_B(K * N), h_C(M * N);
  std::random_device                    rd;
  std::mt19937                          gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto& vec : {&h_A, &h_B}) {
#if 1
    for (auto& data : *vec) {
      data = dis(gen) / (fabs(dis(gen)) + 1.0000);
    }
#else
#if 0
    if (vec == &h_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row            = i / K;
        int col            = i % K;
        vec->operator[](i) = (row == col);
        if (row < limit && col < limit) {
          vec->operator[](i) = (row == col);
        }
        else {
          vec->operator[](i) = 0;
        }
      }
    }
    if (vec == &h_B) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row = i / N;
        int col = i % N;
        if (row < limit && col < limit) {
          vec->operator[](i) = row * limit + col;
        }
        else {
          vec->operator[](i) = 0;
        }
      }
    }
#else
    if (vec == &h_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row            = i / K;
        int col            = i % K;
        vec->operator[](i) = dis(gen);
      }
    }
    if (vec == &h_B) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row            = i / N;
        int col            = i % N;
        vec->operator[](i) = (row == col);
      }
    }
#endif
#endif
  }

  float *d_A, *d_B, *d_B_transposed, *d_C;
  for (auto& pair : {std::make_pair(h_A, &d_A),
                     std::make_pair(h_B, &d_B),
                     std::make_pair(h_B, &d_B_transposed),
                     std::make_pair(h_C, &d_C)}) {
    const std::vector<float>& h      = pair.first;
    float*&                   device = *pair.second;
    CHECK_CUDA_RETURN(cudaMalloc(&device, sizeof(float) * h.size()));
    CHECK_CUDA_RETURN(cudaMemcpy(device, h.data(), sizeof(float) * h.size(), cudaMemcpyDefault));
  }

  {
    cudaMemset(d_C, 0, M * N * sizeof(float));
    launch_fp32_naive_mm(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C.data(), d_C, sizeof(float) * h_C.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  {
    LLMMM::transpose_matrix__aligned_128(d_B_transposed, d_B, M, N, nullptr);
    CHECK_CUDA_ERROR();
  }

  std::vector<float>         h_A_s(M * K / 128);
  float*                     d_A_s;
  std::vector<__nv_fp8_e4m3> h_A_q(M * K);
  __nv_fp8_e4m3*             d_A_q;
  std::vector<float>         h_B_s(N / 128 * K / 128);
  float*                     d_B_s;
  std::vector<__nv_fp8_e4m3> h_B_q(N * K);
  __nv_fp8_e4m3*             d_B_q;
  std::vector<float>         h_C_s(M * N / 128);
  float*                     d_C_s;
  std::vector<__nv_fp8_e4m3> h_C_q(M * N);
  __nv_fp8_e4m3*             d_C_q;
  for (auto& pair : {std::make_pair(h_A_s, &d_A_s), std::make_pair(h_B_s, &d_B_s), std::make_pair(h_C_s, &d_C_s)}) {
    const std::vector<float>& h      = pair.first;
    float*&                   device = *pair.second;
    CHECK_CUDA_RETURN(cudaMalloc(&device, sizeof(float) * h.size()));
  }

  for (auto& pair : {std::make_pair(h_A_q, &d_A_q), std::make_pair(h_B_q, &d_B_q), std::make_pair(h_C_q, &d_C_q)}) {
    const std::vector<__nv_fp8_e4m3>& h      = pair.first;
    __nv_fp8_e4m3*&                   device = *pair.second;
    CHECK_CUDA_RETURN(cudaMalloc(&device, sizeof(__nv_fp8_e4m3) * h.size()));
  }

  fp8_blockwise_symmetric_quantization<1, 128, true>(d_A, d_A_q, d_A_s, M, K, nullptr);
  CHECK_CUDA_ERROR();
  CHECK_CUDA_RETURN(cudaMemcpy(h_A_q.data(), d_A_q, sizeof(*d_A_q) * M * K, cudaMemcpyDefault));
  CHECK_CUDA_RETURN(cudaMemcpy(h_A_s.data(), d_A_s, sizeof(*d_A_s) * M * K / 128, cudaMemcpyDefault));
  {
    float max_abs_error = std::numeric_limits<float>::min();
    float base, exp;
    int   error_m, error_k;
    float q, s;
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        float _q   = float(h_A_q[m * K + k]);
        float _s   = h_A_s[k / 128 * M + m];
        float deq  = _q * _s;
        float fp32 = h_A[m * K + k];
        float diff = fabs(fp32 - deq);
        if (diff > max_abs_error) {
          max_abs_error = diff;
          base          = fp32;
          exp           = deq;
          error_m       = m;
          error_k       = k;
          q             = _q;
          s             = _s;
        }
      }
    }
    printf(
      "check quant A, max_abs_error = %10.8f, max_relative_error = %10.8f, base = %10.8f, exp = %10.8f, q = %10.8f, s = %10.8f, m = %5d, k = %5d\n",
      max_abs_error,
      max_abs_error / fabs(base),
      base,
      exp,
      q,
      s,
      error_m,
      error_k);
  }
  // {
  //   printf("A_q\n");
  //   for (int m = 0; m < M && m < 128; ++m) {
  //     for (int k = 0; k < K && k < 128; ++k) {
  //       printf("%10.3f(%10.3f) ", float(h_A_q[m * K + k]), h_A[m * K + k]);
  //     }
  //     printf("\n");
  //   }
  //   printf("A_s\n");
  //   for (int m = 0; m < 128; ++m) {
  //     printf("%10.8f ", float(h_A_s[m]));
  //   }
  //   printf("\n");
  // }

  fp8_blockwise_symmetric_quantization<128, 128, false>(d_B_transposed, d_B_q, d_B_s, K, N, nullptr);
  CHECK_CUDA_ERROR();
  CHECK_CUDA_RETURN(cudaMemcpy(h_B_q.data(), d_B_q, sizeof(*d_B_q) * N * K, cudaMemcpyDefault));
  CHECK_CUDA_RETURN(cudaMemcpy(h_B_s.data(), d_B_s, sizeof(*d_B_s) * N / 128 * K / 128, cudaMemcpyDefault));
  {
    float max_abs_error = std::numeric_limits<float>::min();
    float base, exp;
    int   error_n, error_k;
    float q, s;
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        float _q   = float(h_B_q[n * K + k]);
        float _s   = h_B_s[n / 128 * (K / 128) + k / 128];
        float deq  = _q * _s;
        float fp32 = h_B[k * N + n];
        float diff = fabs(fp32 - deq);
        if (diff > max_abs_error) {
          max_abs_error = diff;
          base          = fp32;
          exp           = deq;
          error_n       = n;
          error_k       = k;
          q             = _q;
          s             = _s;
        }
      }
    }
    printf(
      "check quant B, max_abs_error = %10.8f, max_relative_error = %10.8f, base = %10.8f, exp = %10.8f, q = %10.8f, s = %10.8f, n = %5d, k = %5d\n",
      max_abs_error,
      max_abs_error / fabs(base),
      base,
      exp,
      q,
      s,
      error_n,
      error_k);
  }

  float*             d_C_fp32;
  std::vector<float> h_C_fp32(M * N);
  CHECK_CUDA_RETURN(cudaMalloc(&d_C_fp32, sizeof(float) * h_C_fp32.size()));
  fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32(
    d_A_q, d_A_s, d_B_q, d_B_s, d_C_fp32, M, N, K, nullptr);
  CHECK_CUDA_ERROR();
  CHECK_CUDA_RETURN(cudaMemcpy(h_C_fp32.data(), d_C_fp32, sizeof(float) * h_C_fp32.size(), cudaMemcpyDefault));
  {
    float max_abs_error = std::numeric_limits<float>::min();
    float base, exp;
    int   error_m, error_n;
    float sum_abs_error = 0;
    float sum_abs_value = 0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float diff = fabs(h_C[m * N + n] - h_C_fp32[m * N + n]);
        if (diff > max_abs_error) {
          max_abs_error = diff;
          base          = h_C[m * M + n];
          exp           = h_C_fp32[m * M + n];
          error_m       = m;
          error_n       = n;
        }
        sum_abs_error += diff;
        sum_abs_value += fabs(h_C[m * M + n]);
      }
    }
    printf(
      "check fp8 mm, max_abs_error = %10.8f, max_relative_error = %10.8f, avg_abs_error = %10.8f, avg_relative_error = %10.8f, base = %10.8f, exp = %10.8f, m = %5d, n = %5d\n",
      max_abs_error,
      max_abs_error / fabs(base),
      sum_abs_error / (M * N),
      sum_abs_error / sum_abs_value,
      base,
      exp,
      error_m,
      error_n);
  }

  CHECK_CUDA_RETURN(cudaMemset(d_C_fp32, 0, sizeof(float) * h_C_fp32.size()));
  fp8_gemm_blockwise_quant_A_1x128__B_128x128__output_fp32__partial_loop_k(
    d_A_q, d_A_s, d_B_q, d_B_s, d_C_fp32, M, N, K, nullptr);
  CHECK_CUDA_ERROR();
  CHECK_CUDA_RETURN(cudaMemcpy(h_C_fp32.data(), d_C_fp32, sizeof(float) * h_C_fp32.size(), cudaMemcpyDefault));
  {
    float max_abs_error = std::numeric_limits<float>::min();
    float base, exp;
    int   error_m, error_n;
    float sum_abs_error = 0;
    float sum_abs_value = 0;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float diff = fabs(h_C[m * N + n] - h_C_fp32[m * N + n]);
        if (diff > max_abs_error) {
          max_abs_error = diff;
          base          = h_C[m * M + n];
          exp           = h_C_fp32[m * M + n];
          error_m       = m;
          error_n       = n;
        }
        sum_abs_error += diff;
        sum_abs_value += fabs(h_C[m * M + n]);
      }
    }
    printf(
      "check fp8 mm, max_abs_error = %10.8f, max_relative_error = %10.8f, avg_abs_error = %10.8f, avg_relative_error = %10.8f, base = %10.8f, exp = %10.8f, m = %5d, n = %5d\n",
      max_abs_error,
      max_abs_error / fabs(base),
      sum_abs_error / (M * N),
      sum_abs_error / sum_abs_value,
      base,
      exp,
      error_m,
      error_n);
  }

#define check(function)                                                                                                                                                                    \
  CHECK_CUDA_RETURN(cudaMemset(d_C_q, 0, sizeof(__nv_fp8_e4m3) * h_C_q.size()));                                                                                                           \
  function(d_A_q, d_A_s, d_B_q, d_B_s, d_C_q, d_C_s, M, N, K, nullptr);                                                                                                                    \
  CHECK_CUDA_ERROR();                                                                                                                                                                      \
  CHECK_CUDA_RETURN(cudaMemcpy(h_C_q.data(), d_C_q, sizeof(__nv_fp8_e4m3) * h_C_q.size(), cudaMemcpyDefault));                                                                             \
  CHECK_CUDA_RETURN(cudaMemcpy(h_C_s.data(), d_C_s, sizeof(float) * h_C_s.size(), cudaMemcpyDefault));                                                                                     \
  {                                                                                                                                                                                        \
    float max_abs_error = std::numeric_limits<float>::min();                                                                                                                               \
    float base, exp_q, exp_s;                                                                                                                                                              \
    int   error_m, error_n;                                                                                                                                                                \
    float sum_abs_error = 0;                                                                                                                                                               \
    float sum_abs_value = 0;                                                                                                                                                               \
    for (int m = 0; m < M; ++m) {                                                                                                                                                          \
      for (int n = 0; n < N; ++n) {                                                                                                                                                        \
        float s    = h_C_s[n / 128 * M + m];                                                                                                                                               \
        float q    = float(h_C_q[m * N + n]);                                                                                                                                              \
        float deq  = q * s;                                                                                                                                                                \
        float diff = fabs(deq - h_C[m * N + n]);                                                                                                                                           \
        if (diff > max_abs_error) {                                                                                                                                                        \
          max_abs_error = diff;                                                                                                                                                            \
          base          = h_C[m * M + n];                                                                                                                                                  \
          exp_q         = q;                                                                                                                                                               \
          exp_s         = s;                                                                                                                                                               \
          error_m       = m;                                                                                                                                                               \
          error_n       = n;                                                                                                                                                               \
        }                                                                                                                                                                                  \
        sum_abs_error += diff;                                                                                                                                                             \
        sum_abs_value += fabs(h_C[m * M + n]);                                                                                                                                             \
      }                                                                                                                                                                                    \
    }                                                                                                                                                                                      \
    printf("\nfunction = %s\n", #function);                                                                                                                                                \
    printf(                                                                                                                                                                                \
      "max_abs_error = %10.8f, max_relative_error = %10.8f, avg_abs_error = %10.8f, avg_relative_error = %10.8f, base = %10.8f, deq = %10.8f, q = %10.8f, s = %10.8f, m = %5d, n = %5d\n", \
      max_abs_error,                                                                                                                                                                       \
      max_abs_error / fabs(base),                                                                                                                                                          \
      sum_abs_error / (M * N),                                                                                                                                                             \
      sum_abs_error / sum_abs_value,                                                                                                                                                       \
      base,                                                                                                                                                                                \
      exp_q * exp_s,                                                                                                                                                                       \
      exp_q,                                                                                                                                                                               \
      exp_s,                                                                                                                                                                               \
      error_m,                                                                                                                                                                             \
      error_n);                                                                                                                                                                            \
  }

  check(fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8);
  check(fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__opt_reg);
#undef check

  CHECK_CUDA_RETURN(cudaFree(d_A));
  CHECK_CUDA_RETURN(cudaFree(d_B));
  CHECK_CUDA_RETURN(cudaFree(d_C));
  CHECK_CUDA_RETURN(cudaFree(d_A_s));
  CHECK_CUDA_RETURN(cudaFree(d_B_s));
  CHECK_CUDA_RETURN(cudaFree(d_C_s));
  CHECK_CUDA_RETURN(cudaFree(d_A_q));
  CHECK_CUDA_RETURN(cudaFree(d_B_q));
  CHECK_CUDA_RETURN(cudaFree(d_C_q));
  CHECK_CUDA_RETURN(cudaFree(d_C_fp32));
  return 0;
}
