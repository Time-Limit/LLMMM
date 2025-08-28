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
__global__ void fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__quadra_buffer__no_sm(
  const __nv_fp8_e4m3* A,
  const float*         A_scale_transposed,
  const __nv_fp8_e4m3* B_transposed,
  const float*         B_scale_transposed,
  __nv_fp8_e4m3*       C,
  float*               C_scale_transposed,
  int                  M,
  int                  N,
  int                  K)
{
  constexpr int M_WARP_COUNT     = BLOCK_M / WARP_M;
  constexpr int N_WARP_COUNT     = BLOCK_N / WARP_N;
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
  constexpr int    LDG_S_REG_BUFFER_SIZE = 2;
  float            A_scale_reg[LDG_S_REG_BUFFER_SIZE][M_GROUP_PER_WARP][2];  // A_scale_transposed is (K/128) x M
  float            B_scale_reg;                                              // B_scale_transposed is (N/128) x (K/128)

  constexpr int CAL_BUFFER_SIZE = 2;
  // MMA
  union {
    float ldg[2];
    fp8_t mma[8];
  } A_cal_reg[CAL_BUFFER_SIZE][M_GROUP_PER_WARP];
  union {
    float ldg[4];
    fp8_t mma[16];
  } B_cal_reg[CAL_BUFFER_SIZE][N_GROUP_PER_WARP];

  float           C_mma_reg[2][4];
  float           C_cal_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4] = {0};
  constexpr float ZERO_ARR[4]                                      = {0, 0, 0, 0};

  enum {
    LDG_OFF = 0,
    LDG_ON_Q,
    LDG_ON_S,
    LDG_ON_S_POST,
  };

  enum {
    CAL_OFF = 0,
    CAL_ON,
  };

#define alternate__per_rank(ldg_switch, cal_switch, ldg_k, ldg_q_idx, cal_reg_idx, rank)                               \
  {                                                                                                                    \
    if constexpr (ldg_switch == LDG_ON_Q && rank < M_GROUP_PER_WARP) {                                                 \
      constexpr int m_group        = rank;                                                                             \
      constexpr int m_group_offset = m_group * 8;                                                                      \
      const int     m_lane_offset  = lane_id / 4;                                                                      \
      const int     k_lane_offset  = lane_id % 4 * 8;                                                                  \
      const int     m_global       = m_block_offset + m_warp_offset + m_group_offset + m_lane_offset;                  \
      const int     k_global       = k_lane_offset + ldg_k;                                                            \
      FETCH_FLOAT2(A_cal_reg[ldg_q_idx][m_group].ldg, A[OFFSET(m_global, k_global, K)]);                               \
    }                                                                                                                  \
    if constexpr (ldg_switch == LDG_ON_Q && M_GROUP_PER_WARP <= rank && rank < M_GROUP_PER_WARP + N_GROUP_PER_WARP) {  \
      constexpr int n_group        = rank - M_GROUP_PER_WARP;                                                          \
      constexpr int n_group_offset = n_group * 16;                                                                     \
      const int     n_lane_offset  = lane_id / 4 + (lane_id & 0x1) * 8;                                                \
      const int     k_lane_offset  = lane_id % 4 / 2 * 16;                                                             \
      const int     n_global       = n_block_offset + n_warp_offset + n_group_offset + n_lane_offset;                  \
      const int     k_global       = k_lane_offset + ldg_k;                                                            \
      FETCH_FLOAT4(B_cal_reg[ldg_q_idx][n_group].ldg, B_transposed[OFFSET(n_global, k_global, K)]);                    \
    }                                                                                                                  \
    if constexpr (ldg_switch == LDG_ON_S && rank == 0) {                                                               \
      FETCH_FLOAT(B_scale_reg, B_scale_transposed[OFFSET(n_block_offset / 128, ldg_k / 128, K / 128)]);                \
    }                                                                                                                  \
    if constexpr (ldg_switch == LDG_ON_S && rank < M_GROUP_PER_WARP) {                                                 \
      constexpr int mg = rank;                                                                                         \
      FETCH_FLOAT2(                                                                                                    \
        A_scale_reg[1][mg],                                                                                            \
        A_scale_transposed[OFFSET(ldg_k / 128, m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2, M)]);        \
    }                                                                                                                  \
    if constexpr (ldg_switch == LDG_ON_S_POST && rank < M_GROUP_PER_WARP) {                                            \
      constexpr int mg      = rank;                                                                                    \
      A_scale_reg[0][mg][0] = A_scale_reg[1][mg][0] * B_scale_reg;                                                     \
      A_scale_reg[0][mg][1] = A_scale_reg[1][mg][1] * B_scale_reg;                                                     \
    }                                                                                                                  \
    if constexpr (cal_switch && rank < MxN_GROUP_PER_WARP) {                                                           \
      constexpr int mg  = rank % M_GROUP_PER_WARP;                                                                     \
      constexpr int ng  = rank / M_GROUP_PER_WARP;                                                                     \
      constexpr int idx = rank % 2;                                                                                    \
      mma_m16n8k32_row_col(C_mma_reg[idx], B_cal_reg[cal_reg_idx][ng].mma, A_cal_reg[cal_reg_idx][mg].mma, ZERO_ARR);  \
    }                                                                                                                  \
    if constexpr (cal_switch && rank > 0 && rank <= MxN_GROUP_PER_WARP) {                                              \
      constexpr int mg  = (rank - 1) % M_GROUP_PER_WARP;                                                               \
      constexpr int ng  = (rank - 1) / M_GROUP_PER_WARP;                                                               \
      constexpr int idx = (rank - 1) % 2;                                                                              \
      C_cal_reg[mg][ng][0] += C_mma_reg[idx][0] * A_scale_reg[0][mg][0];                                               \
      C_cal_reg[mg][ng][1] += C_mma_reg[idx][1] * A_scale_reg[0][mg][1];                                               \
      C_cal_reg[mg][ng][2] += C_mma_reg[idx][2] * A_scale_reg[0][mg][0];                                               \
      C_cal_reg[mg][ng][3] += C_mma_reg[idx][3] * A_scale_reg[0][mg][1];                                               \
    }                                                                                                                  \
  }

#define alternate(ldg_switch, cal_switch, ldg_k, ldg_q_idx, cal_reg_idx)                                               \
  {                                                                                                                    \
    constexpr int MxN_GROUP_PER_WARP = M_GROUP_PER_WARP * N_GROUP_PER_WARP;                                            \
    static_assert(MxN_GROUP_PER_WARP <= 32);                                                                           \
    static_assert(M_GROUP_PER_WARP + N_GROUP_PER_WARP <= 32);                                                          \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 0);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 1);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 2);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 3);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 4);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 5);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 6);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 7);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 8);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 9);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 10);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 11);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 12);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 13);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 14);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 15);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 16);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 17);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 18);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 19);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 20);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 21);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 22);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 23);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 24);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 25);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 26);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 27);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 28);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 29);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 30);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 31);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 32);                              \
  }

  constexpr int IGN  = -1;
  constexpr int IDX0 = 0, IDX1 = 1;

  {
    alternate(LDG_ON_S, CAL_OFF, 0, IGN, IGN);
    alternate(LDG_ON_Q, CAL_OFF, 0, IDX0, IGN);
    alternate(LDG_ON_S_POST, CAL_OFF, IGN, IGN, IGN);
  }

  int k_block_offset = 0;
  static_assert(LOOP_K == 128);
  while (k_block_offset + 128 < K) {
    alternate(LDG_ON_S, CAL_OFF, k_block_offset + 128, IGN, IGN);
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX0);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q, CAL_OFF, k_block_offset, IDX0, IGN);
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX1);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX0);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q, CAL_OFF, k_block_offset, IDX0, IGN);
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX1);
    }
    alternate(LDG_ON_S_POST, CAL_OFF, IGN, IGN, IGN);
  }
  {
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX0);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q, CAL_OFF, k_block_offset, IDX0, IGN);
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX1);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX0);
    }
    {
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX1);
    }
  }

  constexpr int m_lane_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  float         max_val[M_GROUP_PER_WARP];
  __shared__ float C_block_extrema[M_WARP_COUNT][WARP_M][N_WARP_COUNT];

  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    max_val[mg] = fabs(C_cal_reg[mg][0][0]);
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      LLMMM::shfl_1_and_0(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_3_and_2(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_23_and_01(C_cal_reg[mg][ng], 0x8, lane_id);

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
    const float scale_inv = fp8_e4m3_range / max_val[mg];
    const float scale     = max_val[mg] / fp8_e4m3_range;
    const int   m_global  = m_block_offset + m_warp_offset + mg * 8 + m_lane_offset[lane_id % 8];
    static_assert(N_GROUP_PER_WARP % 2 == 0);
    for (int ng = 0; ng < N_GROUP_PER_WARP; ng += 2) {
      const int n_global = n_block_offset + n_warp_offset + ng * 16 + lane_id / 8 * 8;
      fp8_t     q[8]     = {
        fp8_t(C_cal_reg[mg][ng][0] * scale_inv),
        fp8_t(C_cal_reg[mg][ng][1] * scale_inv),
        fp8_t(C_cal_reg[mg][ng][2] * scale_inv),
        fp8_t(C_cal_reg[mg][ng][3] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][0] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][1] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][2] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][3] * scale_inv),
      };
      LLMMM::shfl_4567_and_0123(q, 0x10, lane_id);
      STORE_FLOAT2(C[OFFSET(m_global, n_global, N)], q);
    }
    if (lane_id < 8) {
      static_assert(BLOCK_N <= 128);
      STORE_FLOAT(C_scale_transposed[OFFSET(n_block_offset / 128, m_global, M)], scale);
    }
  }
#undef alternate
#undef alternate__per_rank
}

template<int BLOCK_M, int BLOCK_N, int WARP_M, int WARP_N, int LOOP_K>
__global__ void
fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__quadra_buffer__no_sm__B_Q_single_buffer(
  const __nv_fp8_e4m3* A,
  const float*         A_scale_transposed,
  const __nv_fp8_e4m3* B_transposed,
  const float*         B_scale_transposed,
  __nv_fp8_e4m3*       C,
  float*               C_scale_transposed,
  int                  M,
  int                  N,
  int                  K)
{
  constexpr int M_WARP_COUNT     = BLOCK_M / WARP_M;
  constexpr int N_WARP_COUNT     = BLOCK_N / WARP_N;
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
  constexpr int    LDG_S_REG_BUFFER_SIZE = 2;
  float            A_scale_reg[LDG_S_REG_BUFFER_SIZE][M_GROUP_PER_WARP][2];  // A_scale_transposed is (K/128) x M
  float            B_scale_reg;                                              // B_scale_transposed is (N/128) x (K/128)

  constexpr int CAL_BUFFER_SIZE = 2;
  // MMA
  union {
    float ldg[2];
    fp8_t mma[8];
  } A_cal_reg[CAL_BUFFER_SIZE][M_GROUP_PER_WARP];

  union {
    float ldg[4];
    fp8_t mma[16];
  } B_cal_reg[N_GROUP_PER_WARP];

  float           C_mma_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4] = {0};
  float           C_cal_reg[M_GROUP_PER_WARP][N_GROUP_PER_WARP][4] = {0};
  constexpr float ZERO_ARR[4]                                      = {0, 0, 0, 0};

  enum {
    LDG_OFF = 0,
    LDG_ON_Q_A,
    LDG_ON_Q_B,
    LDG_ON_S,
    LDG_ON_S_POST,
  };

  enum {
    CAL_OFF = 0,
    CAL_ON,
  };

  const fp8_t* A_partial_ptr[M_GROUP_PER_WARP];
  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    A_partial_ptr[mg] = &A[(m_block_offset + m_warp_offset + mg * 8 + lane_id / 4) * K + lane_id % 4 * 8];
  }
  const fp8_t* B_parital_ptr[N_GROUP_PER_WARP];
  for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
    B_parital_ptr[ng] = &B_transposed[(n_block_offset + n_warp_offset + ng * 16 + lane_id / 4 + (lane_id & 0x1) * 8) * K
                                      + lane_id % 4 / 2 * 16];
  }

  const float* B_scale_partial_ptr = &B_scale_transposed[(n_block_offset / 128) * (K / 128)];

  const float* A_scale_partial_ptr[M_GROUP_PER_WARP];
  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    A_scale_partial_ptr[mg] = &A_scale_transposed[m_block_offset + m_warp_offset + mg * 8 + lane_id % 4 * 2];
}

#define alternate__per_rank(ldg_switch, cal_switch, ldg_k, ldg_q_idx, cal_reg_idx, rank)                               \
{                                                                                                                      \
  if constexpr (cal_switch && rank > 0 && rank <= MxN_GROUP_PER_WARP) {                                                \
    constexpr int mg = (rank - 1) % M_GROUP_PER_WARP;                                                                  \
    constexpr int ng = (rank - 1) / M_GROUP_PER_WARP;                                                                  \
    C_cal_reg[mg][ng][0] += C_mma_reg[mg][ng][0] * A_scale_reg[0][mg][0];                                              \
    C_cal_reg[mg][ng][1] += C_mma_reg[mg][ng][1] * A_scale_reg[0][mg][1];                                              \
    C_cal_reg[mg][ng][2] += C_mma_reg[mg][ng][2] * A_scale_reg[0][mg][0];                                              \
    C_cal_reg[mg][ng][3] += C_mma_reg[mg][ng][3] * A_scale_reg[0][mg][1];                                              \
  }                                                                                                                    \
  if constexpr (cal_switch && rank < MxN_GROUP_PER_WARP) {                                                             \
    constexpr int mg = rank % M_GROUP_PER_WARP;                                                                        \
    constexpr int ng = rank / M_GROUP_PER_WARP;                                                                        \
    mma_m16n8k32_row_col(C_mma_reg[mg][ng], B_cal_reg[ng].mma, A_cal_reg[cal_reg_idx][mg].mma, ZERO_ARR);              \
  }                                                                                                                    \
  if constexpr (ldg_switch == LDG_ON_Q_A && rank < M_GROUP_PER_WARP) {                                                 \
    constexpr int m_group = rank;                                                                                      \
    /* constexpr int m_group_offset = m_group * 8;                                                      */             \
    /* const int     m_lane_offset  = lane_id / 4;                                                      */             \
    /* const int     k_lane_offset  = lane_id % 4 * 8;                                                  */             \
    /* const int     m_global       = m_block_offset + m_warp_offset + m_group_offset + m_lane_offset;  */             \
    /* const int     k_global       = k_lane_offset + ldg_k;                                            */             \
    FETCH_FLOAT2_WITH_SRC_PTR(A_cal_reg[ldg_q_idx][m_group].ldg, A_partial_ptr[m_group] + ldg_k);                      \
  }                                                                                                                    \
  if constexpr (ldg_switch == LDG_ON_Q_B && rank <= MxN_GROUP_PER_WARP && rank % M_GROUP_PER_WARP == 0                 \
                && rank / M_GROUP_PER_WARP > 0) {                                                                      \
    constexpr int n_group = rank / M_GROUP_PER_WARP - 1;                                                               \
    /* constexpr int n_group_offset = n_group * 16;                                                       */           \
    /* const int     n_lane_offset  = lane_id / 4 + (lane_id & 0x1) * 8;                                  */           \
    /* const int     k_lane_offset  = lane_id % 4 / 2 * 16;                                               */           \
    /* const int     n_global       = n_block_offset + n_warp_offset + n_group_offset + n_lane_offset;    */           \
    /* const int     k_global       = k_lane_offset + ldg_k;                                              */           \
    FETCH_FLOAT4_WITH_SRC_PTR(B_cal_reg[n_group].ldg, B_parital_ptr[n_group] + ldg_k);                                 \
  }                                                                                                                    \
  if constexpr (ldg_switch == LDG_ON_S && rank == 0) {                                                                 \
    FETCH_FLOAT(B_scale_reg, *(B_scale_partial_ptr + ldg_k / 128));                                                    \
  }                                                                                                                    \
  if constexpr (ldg_switch == LDG_ON_S && rank < M_GROUP_PER_WARP) {                                                   \
    constexpr int mg = rank;                                                                                           \
    FETCH_FLOAT2_WITH_SRC_PTR(A_scale_reg[1][mg], A_scale_partial_ptr[mg] + ldg_k / 128 * M);                          \
  }                                                                                                                    \
  if constexpr (ldg_switch == LDG_ON_S_POST && rank < M_GROUP_PER_WARP) {                                              \
    constexpr int mg      = rank;                                                                                      \
    A_scale_reg[0][mg][0] = A_scale_reg[1][mg][0] * B_scale_reg;                                                       \
    A_scale_reg[0][mg][1] = A_scale_reg[1][mg][1] * B_scale_reg;                                                       \
  }                                                                                                                    \
}

#define alternate(ldg_switch, cal_switch, ldg_k, ldg_q_idx, cal_reg_idx)                                               \
  {                                                                                                                    \
    constexpr int MxN_GROUP_PER_WARP = M_GROUP_PER_WARP * N_GROUP_PER_WARP;                                            \
    static_assert(MxN_GROUP_PER_WARP <= 32);                                                                           \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 0);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 1);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 2);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 3);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 4);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 5);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 6);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 7);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 8);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 9);                               \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 10);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 11);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 12);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 13);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 14);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 15);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 16);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 17);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 18);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 19);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 20);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 21);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 22);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 23);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 24);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 25);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 26);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 27);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 28);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 29);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 30);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 31);                              \
    alternate__per_rank(ldg_switch, cal_switch, (ldg_k), (ldg_q_idx), (cal_reg_idx), 32);                              \
  }

  constexpr int IGN  = -1;
  constexpr int IDX0 = 0, IDX1 = 1;

  {
    alternate(LDG_ON_S, CAL_OFF, 0, IGN, IGN);
    alternate(LDG_ON_Q_A, CAL_OFF, 0, IDX0, IGN);
    alternate(LDG_ON_Q_B, CAL_OFF, 0, IGN, IGN);
    alternate(LDG_ON_S_POST, CAL_OFF, IGN, IGN, IGN);
  }

  int k_block_offset = 0;
  static_assert(LOOP_K == 128);
  while (k_block_offset + 128 < K) {
    alternate(LDG_ON_S, CAL_OFF, k_block_offset + 128, IGN, IGN);
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q_A, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_ON_Q_B, CAL_ON, k_block_offset, IGN, IDX0);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q_A, CAL_OFF, k_block_offset, IDX0, IGN);
      alternate(LDG_ON_Q_B, CAL_ON, k_block_offset, IGN, IDX1);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q_A, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_ON_Q_B, CAL_ON, k_block_offset, IGN, IDX0);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q_A, CAL_OFF, k_block_offset, IDX0, IGN);
      alternate(LDG_ON_Q_B, CAL_ON, k_block_offset, IGN, IDX1);
    }
    alternate(LDG_ON_S_POST, CAL_OFF, IGN, IGN, IGN);
  }
  {
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q_A, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_ON_Q_B, CAL_ON, k_block_offset, IGN, IDX0);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q_A, CAL_OFF, k_block_offset, IDX0, IGN);
      alternate(LDG_ON_Q_B, CAL_ON, k_block_offset, IGN, IDX1);
    }
    {
      k_block_offset += 32;
      alternate(LDG_ON_Q_A, CAL_OFF, k_block_offset, IDX1, IGN);
      alternate(LDG_ON_Q_B, CAL_ON, k_block_offset, IGN, IDX0);
    }
    {
      alternate(LDG_OFF, CAL_ON, IGN, IGN, IDX1);
    }
  }

  constexpr int m_lane_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  float         max_val[M_GROUP_PER_WARP];
  __shared__ float C_block_extrema[M_WARP_COUNT][WARP_M][N_WARP_COUNT];

  for (int mg = 0; mg < M_GROUP_PER_WARP; ++mg) {
    max_val[mg] = fabs(C_cal_reg[mg][0][0]);
    for (int ng = 0; ng < N_GROUP_PER_WARP; ++ng) {
      LLMMM::shfl_1_and_0(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_3_and_2(C_cal_reg[mg][ng], 0x4, lane_id);
      LLMMM::shfl_23_and_01(C_cal_reg[mg][ng], 0x8, lane_id);

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
    const float scale_inv = fp8_e4m3_range / max_val[mg];
    const float scale     = max_val[mg] / fp8_e4m3_range;
    const int   m_global  = m_block_offset + m_warp_offset + mg * 8 + m_lane_offset[lane_id % 8];
    static_assert(N_GROUP_PER_WARP % 2 == 0);
    for (int ng = 0; ng < N_GROUP_PER_WARP; ng += 2) {
      const int n_global = n_block_offset + n_warp_offset + ng * 16 + lane_id / 8 * 8;
      fp8_t     q[8]     = {
        fp8_t(C_cal_reg[mg][ng][0] * scale_inv),
        fp8_t(C_cal_reg[mg][ng][1] * scale_inv),
        fp8_t(C_cal_reg[mg][ng][2] * scale_inv),
        fp8_t(C_cal_reg[mg][ng][3] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][0] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][1] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][2] * scale_inv),
        fp8_t(C_cal_reg[mg][ng + 1][3] * scale_inv),
      };
      LLMMM::shfl_4567_and_0123(q, 0x10, lane_id);
      STORE_FLOAT2(C[OFFSET(m_global, n_global, N)], q);
    }
    if (lane_id < 8) {
      static_assert(BLOCK_N <= 128);
      STORE_FLOAT(C_scale_transposed[OFFSET(n_block_offset / 128, m_global, M)], scale);
    }
  }
#undef alternate
#undef alternate__per_rank
}

#define launch_128x128_64x64(function)                                                                                 \
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
    constexpr int WARP_M  = 64;                                                                                        \
    constexpr int WARP_N  = 64;                                                                                        \
    static_assert(BLOCK_M > 0 && BLOCK_M <= 128 && BLOCK_M % WARP_M == 0);                                             \
    static_assert(BLOCK_N == 128 && BLOCK_N % WARP_N == 0);                                                            \
    static_assert(WARP_M > 0 && WARP_M % 8 == 0); /* mma.m16n8k32.row.col, A is n8k32, B is m16k32 */                  \
    static_assert(WARP_N > 0 && WARP_N % 16 == 0);                                                                     \
    static_assert(LOOP_K == 128);                                                                                      \
    constexpr int WARP_COUNT = BLOCK_M / WARP_M * BLOCK_N / WARP_N;                                                    \
    static_assert(0 < WARP_COUNT && WARP_COUNT <= 8 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);                          \
    if (!(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % LOOP_K == 0)) {                                                  \
      throw std::runtime_error("M or N or K are not aligned.");                                                        \
    }                                                                                                                  \
    dim3 grid(N / BLOCK_N, M / BLOCK_M);                                                                               \
    dim3 block(WARP_COUNT * 32);                                                                                       \
    auto kSmemSize   = 0;                                                                                              \
    auto kernel_func = &function<BLOCK_M, BLOCK_N, WARP_M, WARP_N, LOOP_K>;                                            \
    CHECK_CUDA_RETURN(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));      \
    function<BLOCK_M, BLOCK_N, WARP_M, WARP_N, LOOP_K><<<grid, block, 0, stream>>>(                                    \
      A, A_scale_transposed, B_transposed, B_scale_transposed, C, C_scale_transposed, M, N, K);                        \
  }

launch_128x128_64x64(fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__quadra_buffer__no_sm);
launch_128x128_64x64(
  fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__quadra_buffer__no_sm__B_Q_single_buffer);

#undef launch_128x128_64x64

int main()
{
  static const int M = (1 << 12), N = (1 << 12), K = (1 << 12);

  std::vector<float>                    h_A(M * K), h_B(K * N), h_C(M * N);
  std::random_device                    rd;
  std::mt19937                          gen(0);
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

  std::vector<__nv_fp8_e4m3> h_A_q_constructed(M * K);
  __nv_fp8_e4m3*             d_A_q_constructed;

  std::vector<__nv_fp8_e4m3> h_B_q_constructed(N * K);
  __nv_fp8_e4m3*             d_B_q_constructed;

  CHECK_CUDA_RETURN(cudaMalloc(&d_A_q_constructed, sizeof(__nv_fp8_e4m3) * M * K));
  CHECK_CUDA_RETURN(cudaMalloc(&d_B_q_constructed, sizeof(__nv_fp8_e4m3) * N * K));

  construct_m16n8k32_B_layout(d_A_q_constructed, d_A_q, M, K, nullptr);
  CHECK_CUDA_ERROR();
  CHECK_CUDA_RETURN(
    cudaMemcpy(h_A_q_constructed.data(), d_A_q_constructed, sizeof(__nv_fp8_e4m3) * M * K, cudaMemcpyDefault));
  {
    int error = 0;
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        float         base                         = float(h_A_q[OFFSET(m, k, K)]);
        int           construct_k                  = k % 32;
        constexpr int construct_k_group_mapping[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        construct_k                                = construct_k_group_mapping[construct_k / 4] * 4 + construct_k % 4;
        float exp                                  = float(h_A_q_constructed[OFFSET(m, k / 32 * 32 + construct_k, K)]);
        if (base != exp) {
          error++;
        }
      }
    }
    printf("check construct_m16n8k32_B_layout, total = %10d, error = %10d\n", M * K, error);
    // printf("construct_m16n8k32_B_layout, base\n");
    // for (int m = 0; m < M && m < 8; ++m) {
    //   for (int k = 0; k < K && k < 32; ++k) {
    //     printf("%10.3f(%03d, %03d) ", float(h_A_q[OFFSET(m, k, K)]), m, k);
    //     if ((k + 1) % 8 == 0) {
    //       printf("\n");
    //     }
    //   }
    // }
    // printf("construct_m16n8k32_B_layout, exp\n");
    // for (int m = 0; m < M && m < 8; ++m) {
    //   for (int k = 0; k < K && k < 32; ++k) {
    //     printf("%10.3f(%03d, %03d) ", float(h_A_q_constructed[OFFSET(m, k, K)]), m, k);
    //     if ((k + 1) % 8 == 0) {
    //       printf("\n");
    //     }
    //   }
    // }
  }
  construct_m16n8k32_A_layout(d_B_q_constructed, d_B_q, N, K, nullptr);
  CHECK_CUDA_ERROR();
  CHECK_CUDA_RETURN(
    cudaMemcpy(h_B_q_constructed.data(), d_B_q_constructed, sizeof(__nv_fp8_e4m3) * N * K, cudaMemcpyDefault));
  {
    int error = 0;
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        float         base                   = float(h_B_q[OFFSET(n, k, K)]);
        constexpr int construct_n_mapping[4] = {0, 8, 0, 8};
        constexpr int construct_k_mapping[4] = {0, 0, 16, 16};

        int cn = n / 16 * 16 + construct_n_mapping[k % 16 / 4] + n % 8;
        int ck =
          k / 32 * 32 + construct_k_mapping[k % 16 / 4] + (((n % 16) < 8) ? 0 : 4) + (((k % 32) < 16) ? 0 : 8) + k % 4;
        float exp = float(h_B_q_constructed[OFFSET(cn, ck, K)]);
        if (base != exp) {
          error++;
        }
      }
    }
    printf("check construct_m16n8k32_A_layout, total = %10d, error = %10d\n", N * K, error);
    // printf("construct_m16n8k32_A_layout, base\n");
    // for (int n = 0; n < N && n < 16; ++n) {
    //   for (int k = 0; k < K && k < 32; ++k) {
    //     printf("%10.3f(%03d, %03d) ", float(h_B_q[OFFSET(n, k, K)]), n, k);
    //     if ((k + 1) % 8 == 0) {
    //       printf("\n");
    //     }
    //   }
    // }
    // printf("construct_m16n8k32_A_layout, exp\n");
    // for (int n = 0; n < N && n < 16; ++n) {
    //   for (int k = 0; k < K && k < 32; ++k) {
    //     printf("%10.3f(%03d, %03d) ", float(h_B_q_constructed[OFFSET(n, k, K)]), n, k);
    //     if ((k + 1) % 8 == 0) {
    //       printf("\n");
    //     }
    //   }
    // }
  }

#define check(function)                                                                                                                                                                    \
  CHECK_CUDA_RETURN(cudaMemset(d_C_q, 0, sizeof(__nv_fp8_e4m3) * h_C_q.size()));                                                                                                           \
  function(d_A_q_constructed, d_A_s, d_B_q_constructed, d_B_s, d_C_q, d_C_s, M, N, K, nullptr);                                                                                            \
  CHECK_CUDA_ERROR_WITH_INFO(#function);                                                                                                                                                   \
  CHECK_CUDA_RETURN(cudaMemcpy(h_C_q.data(), d_C_q, sizeof(__nv_fp8_e4m3) * h_C_q.size(), cudaMemcpyDefault));                                                                             \
  CHECK_CUDA_RETURN(cudaMemcpy(h_C_s.data(), d_C_s, sizeof(float) * h_C_s.size(), cudaMemcpyDefault));                                                                                     \
  {                                                                                                                                                                                        \
    float max_abs_error = std::numeric_limits<float>::min();                                                                                                                               \
    float base, exp_q, exp_s;                                                                                                                                                              \
    int   error_m, error_n;                                                                                                                                                                \
    float sum_abs_error = 0;                                                                                                                                                               \
    float sum_abs_value = 0;                                                                                                                                                               \
    for (int m = 0; m < M && m < 16; ++m) {                                                                                                                                                \
      for (int n = 0; n < N && n < 8; ++n) {                                                                                                                                               \
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
    /* printf("function = %s, base\n", #function);*/                                                                                                                                       \
    /* for (int m = 0; m < M && m < 16; ++m) {    */                                                                                                                                       \
    /*   for (int n = 0; n < N && n < 8; ++n) {   */                                                                                                                                       \
    /*     printf("%8.3f ", h_C[m * N + n]);      */                                                                                                                                       \
    /*   }                                        */                                                                                                                                       \
    /*   printf("\n");                            */                                                                                                                                       \
    /* }                                          */                                                                                                                                       \
    /* printf("function = %s, q\n", #function);   */                                                                                                                                       \
    /* for (int m = 0; m < M && m < 16; ++m) {    */                                                                                                                                       \
    /*   for (int n = 0; n < N && n < 8; ++n) {   */                                                                                                                                       \
    /*     float q = float(h_C_q[m * N + n]);     */                                                                                                                                       \
    /*     printf("%8.3f ", q);                   */                                                                                                                                       \
    /*   }                                        */                                                                                                                                       \
    /*   printf("\n");                            */                                                                                                                                       \
    /* }                                          */                                                                                                                                       \
  }

  check(fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__quadra_buffer__no_sm);
  check(fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128__output_fp8__quadra_buffer__no_sm__B_Q_single_buffer);
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
  CHECK_CUDA_RETURN(cudaFree(d_A_q_constructed));
  CHECK_CUDA_RETURN(cudaFree(d_B_q_constructed));
  return 0;
}
