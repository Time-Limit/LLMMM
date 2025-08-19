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

#include "util/macro.h"
#include "util/util.cuh"

constexpr int limit = 128;

__global__ void test_fp8_peak_work()
{
  __nv_fp8_e4m3 A[16];
  for (int i = 0; i < 16; ++i) {
    A[i] = __nv_fp8_e4m3(i);
  }
  __nv_fp8_e4m3 B[8];
  for (int i = 0; i < 8; ++i) {
    B[i] = __nv_fp8_e4m3(i);
  }
  float C[4];
  for (int i = 0; i < 20480; ++i) {
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
    LLMMM::mma_sync_aligned_m16n8k32_row_col_f32_f16_f16_f32(C, A, B, C);
  }
  if (C[0] == -1 && C[1] == -1 && C[2] == -1 && C[3] == -1) {
    LLMMM::print_thread_info("");
  }
}

void test_fp8_peak_work(cudaStream_t stream) {
  test_fp8_peak_work<<<512, 1024, 0, stream>>>();
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
        max     = LLMMM::warp_reduce_max(max);
        max     = LLMMM::warp_broadcast(0, max);
        float s = max / fp8_e4m3_range;

        __nv_fp8_e4m3 quanted[4] = {
          __nv_fp8_e4m3(data[0] / s),
          __nv_fp8_e4m3(data[1] / s),
          __nv_fp8_e4m3(data[2] / s),
          __nv_fp8_e4m3(data[3] / s),
        };

        max = max_float;

        STORE_FLOAT(q[OFFSET(m, n_block_offset + lane_id * 4, N)], quanted);
        static_assert(BLOCK_N == QUANT_N);
        if (lane_id == 0) {
          if constexpr (SCALE_TRANPOSE) {
            STORE_FLOAT(scale[OFFSET(n_block_offset / QUANT_N, m, M)], s);
          } else {
          STORE_FLOAT(scale[OFFSET(m, n_block_offset / QUANT_N, scale_N)], s);
          }
        }
      }
    }
  }
  if constexpr (QUANT_M == 128) {
    __shared__ float block_max_value[WARP_COUNT];
    max = LLMMM::warp_reduce_max(max);
    if (lane_id == 0) {
      static_assert(WARP_COUNT == 4);
      block_max_value[warp_id] = max;
    }
    __syncthreads();
    float max4[4];
    FETCH_FLOAT4(max4[0], block_max_value[0]);
    max = (max4[0] > max4[1]) ? max4[0] : max4[1];
    max = (max > max4[2]) ? max : max4[2];
    max = (max > max4[3]) ? max : max4[3];
    float s = max / fp8_e4m3_range;
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
      int m = m_block_offset + loop * WARP_COUNT + warp_id;
      if (m < M) {
        float data[4];
        FETCH_FLOAT4(data[0], x_sm[OFFSET(loop * WARP_COUNT + warp_id, lane_id * 4, BLOCK_N)]);

        __nv_fp8_e4m3 quanted[4] = {
          __nv_fp8_e4m3(data[0] / s),
          __nv_fp8_e4m3(data[1] / s),
          __nv_fp8_e4m3(data[2] / s),
          __nv_fp8_e4m3(data[3] / s),
        };
        STORE_FLOAT(q[OFFSET(m_block_offset + loop * WARP_COUNT + warp_id, n_block_offset + lane_id * 4, N)], quanted);
      }
      static_assert(BLOCK_N == QUANT_N);
      if (threadIdx.x == 0) {
        if constexpr (SCALE_TRANPOSE) {
          STORE_FLOAT(
            scale[OFFSET(n_block_offset / QUANT_N, (m_block_offset + loop * WARP_COUNT + warp_id) / QUANT_M, M)], s);
        } else {
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
      data = dis(gen);
    }
#else
    if (vec == &h_A) {
      for (size_t i = 0; i < vec->size(); ++i) {
        int row            = i / K;
        int col            = i % K;
        vec->operator[](i) = (row == col);
        if (row < limit && col < limit) {
          vec->operator[](i) = row * limit + col;
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
#endif
  }

  float *A, *B, *C;
  for (auto& pair : {std::make_pair(h_A, &A), std::make_pair(h_B, &B), std::make_pair(h_C, &C)}) {
    const std::vector<float>& h      = pair.first;
    float*&                   device = *pair.second;
    CHECK_CUDA_RETURN(cudaMalloc(&device, sizeof(float) * h.size()));
    CHECK_CUDA_RETURN(cudaMemcpy(device, h.data(), sizeof(float) * h.size(), cudaMemcpyDefault));
  }

  std::vector<float>         h_A_s(M * K / 128);
  float*                     d_A_s;
  std::vector<float>         h_B_s(N / 128 * K / 128);
  float*                     d_B_s;
  std::vector<__nv_fp8_e4m3> h_A_q(M * K);
  __nv_fp8_e4m3*             d_A_q;
  std::vector<__nv_fp8_e4m3> h_B_q(N * K);
  __nv_fp8_e4m3*             d_B_q;
  for (auto& pair : {std::make_pair(h_A_s, &d_A_s), std::make_pair(h_B_s, &d_B_s)}) {
    const std::vector<float>& h      = pair.first;
    float*&                   device = *pair.second;
    CHECK_CUDA_RETURN(cudaMalloc(&device, sizeof(float) * h.size()));
  }
  for (auto& pair : {std::make_pair(h_A_q, &d_A_q), std::make_pair(h_B_q, &d_B_q)}) {
    const std::vector<__nv_fp8_e4m3>& h      = pair.first;
    __nv_fp8_e4m3*&                   device = *pair.second;
    CHECK_CUDA_RETURN(cudaMalloc(&device, sizeof(__nv_fp8_e4m3) * h.size()));
  }

  fp8_blockwise_symmetric_quantization<1, 128, true>(A, d_A_q, d_A_s, M, N, nullptr);
  CHECK_CUDA_ERROR();

  CHECK_CUDA_RETURN(
    cudaMemcpy(h_A_q.data(), d_A_q, sizeof(__nv_fp8_e4m3) * h_A_q.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CHECK_CUDA_RETURN(
    cudaMemcpy(h_A_s.data(), d_A_s, sizeof(float) * h_A_s.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
  {
    float max_error = std::numeric_limits<float>::min(), no_quant, quant, scale;
    int   error_m, error_k;
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        float _no_quant = h_A[m * K + k];
        float _quant    = float(h_A_q[m * K + k]);
        float _scale    = h_A_s[k / 128 * M + m];
        if (std::abs(_no_quant) - std::abs(_quant * _scale) > max_error) {
          max_error = std::abs(_no_quant) - std::abs(_quant * _scale);
          no_quant  = _no_quant;
          quant     = _quant;
          scale     = _scale;
          error_m   = m;
          error_k   = k;
        }
      }
    }
    printf(
      "max_absolute_error = %10.8f, max_relative_error = %10.8f, no_quant = %10.8f, quant = %10.8f, scale = %10.8f, dequant = %10.8f, error_m = %04d, error_k = %04d\n",
      max_error,
      max_error / std::abs(no_quant),
      no_quant,
      quant,
      scale,
      scale * quant,
      error_m,
      error_k);
  }

  fp8_blockwise_symmetric_quantization<128, 128, false>(B, d_B_q, d_B_s, N, K, nullptr);
  CHECK_CUDA_ERROR();

  CHECK_CUDA_RETURN(
    cudaMemcpy(h_B_q.data(), d_B_q, sizeof(__nv_fp8_e4m3) * h_B_q.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
  CHECK_CUDA_RETURN(
    cudaMemcpy(h_B_s.data(), d_B_s, sizeof(float) * h_B_s.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));

  {
    float max_error = std::numeric_limits<float>::min(), no_quant, quant, scale;
    int   error_n, error_k;
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        float _no_quant = h_B[k * N + n];
        float _quant    = float(h_B_q[k * N + n]);
        float _scale    = h_B_s[(k / 128) * (N / 128) + n / 128];
        if (std::abs(_no_quant) - std::abs(_quant * _scale) > max_error) {
          max_error = std::abs(_no_quant) - std::abs(_quant * _scale);
          no_quant  = _no_quant;
          quant     = _quant;
          scale     = _scale;
          error_n   = n;
          error_k   = k;
        }
      }
    }
    printf(
      "max_absolute_error = %10.8f, max_relative_error = %10.8f, no_quant = %10.8f, quant = %10.8f, scale = %10.8f, dequant = %10.8f, error_k = %04d, error_n = %04d\n",
      max_error,
      max_error / std::abs(no_quant),
      no_quant,
      quant,
      scale,
      scale * quant,
      error_k,
      error_n);
  }

  test_fp8_peak_work(nullptr);
  CHECK_CUDA_ERROR();

  CHECK_CUDA_RETURN(cudaFree(A));
  CHECK_CUDA_RETURN(cudaFree(B));
  CHECK_CUDA_RETURN(cudaFree(C));
  CHECK_CUDA_RETURN(cudaFree(d_A_s));
  CHECK_CUDA_RETURN(cudaFree(d_B_s));
  CHECK_CUDA_RETURN(cudaFree(d_A_q));
  CHECK_CUDA_RETURN(cudaFree(d_B_q));
  return 0;
}
