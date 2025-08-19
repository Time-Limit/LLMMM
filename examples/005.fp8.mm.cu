#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <random>
#include <vector>

#include "base/base.h"
#include "util/macro.h"
#include "util/util.cuh"

using namespace LLMMM;

constexpr int limit = 128;

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

template<int BLOCK_M, int BLOCK_N, int WARP_M, int WARP_N, int LOOP_K, bool C_SCALE_TRANSPOSE>
__global__ void fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128(const __nv_fp8_e4m3* A,
                                                                     const float*         A_scale_transposed,
                                                                     const __nv_fp8_e4m3* B_transposed,
                                                                     const float*         B_scale_transposed,
                                                                     half*                C,
                                                                     float*               C_scale,
                                                                     int                  M,
                                                                     int                  N,
                                                                     int                  K)
{
}

template<bool C_SCALE_TRANSPOSE>
void fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128(const __nv_fp8_e4m3* A,
                                                          const float*         A_scale_transposed,
                                                          const __nv_fp8_e4m3* B_transposed,
                                                          const float*         B_scale_transposed,
                                                          half*                C,
                                                          float*               C_scale,
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
  static_assert(BLOCK_M % WARP_M == 0);
  static_assert(BLOCK_N % WARP_N == 0);
  static_assert(WARP_M % 16 == 0);
  static_assert(WARP_N % 8 == 0);
  static_assert(LOOP_K == 128);
  constexpr int WARP_COUNT = BLOCK_M / WARP_M * BLOCK_N / WARP_N;
  static_assert(0 < WARP_COUNT && WARP_COUNT <= 4 && (WARP_COUNT & (WARP_COUNT - 1)) == 0);
  if (!(M % BLOCK_M == 0 && N % BLOCK_N == 0 && K % LOOP_K == 0)) {
    throw std::runtime_error("M or N or K are not aligned.");
  }
  dim3 grid(N / BLOCK_N, M / BLOCK_M);
  dim3 block(WARP_COUNT * 32);
  fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128<BLOCK_M, BLOCK_N, WARP_M, WARP_N, LOOP_K, C_SCALE_TRANSPOSE>
    <<<grid, block, 0, stream>>>(A, A_scale_transposed, B_transposed, B_scale_transposed, C, C_scale, M, N, K);
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
    max = warp_reduce_max(max);
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
    transpose_matrix__aligned_128(d_B_transposed, d_B, M, N, nullptr);
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

  fp8_blockwise_symmetric_quantization<1, 128, true>(d_A, d_A_q, d_A_s, M, N, nullptr);
  CHECK_CUDA_ERROR();

  fp8_blockwise_symmetric_quantization<128, 128, false>(d_B_transposed, d_B_q, d_B_s, K, N, nullptr);
  CHECK_CUDA_ERROR();

  half*             d_C_fp16;
  std::vector<half> h_C_fp16(M * N);
  CHECK_CUDA_RETURN(cudaMalloc(&d_C_fp16, sizeof(half) * h_C_fp16.size()));
  fp8_gemm_blockwise_quant_A_1x128__B_128x128__C_1x128<true>(
    d_A_q, d_A_s, d_B_q, d_B_s, d_C_fp16, d_C_s, M, N, K, nullptr);
  CHECK_CUDA_ERROR();

  CHECK_CUDA_RETURN(cudaFree(d_A));
  CHECK_CUDA_RETURN(cudaFree(d_B));
  CHECK_CUDA_RETURN(cudaFree(d_C));
  CHECK_CUDA_RETURN(cudaFree(d_A_s));
  CHECK_CUDA_RETURN(cudaFree(d_B_s));
  CHECK_CUDA_RETURN(cudaFree(d_C_s));
  CHECK_CUDA_RETURN(cudaFree(d_A_q));
  CHECK_CUDA_RETURN(cudaFree(d_B_q));
  CHECK_CUDA_RETURN(cudaFree(d_C_q));
  CHECK_CUDA_RETURN(cudaFree(d_C_fp16));
  return 0;
}
