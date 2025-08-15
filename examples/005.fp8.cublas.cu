#include "util/macro.h"
#include <cmath>
#include <cstdlib>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

void fp8Gemm(const int   m,
             const int   n,
             const int   k,
             const void* A,
             const int   lda,
             const void* B,
             const int   ldb,
             void*       C,
             const int   ldc,
             float*      a_scale,
             float*      b_scale,
             bool        fastAccum)
{
  cublasLtHandle_t cublaslt_handle;
  cublasLtCreate(&cublaslt_handle);
  int                    batch_count   = 1;
  cublasLtMatmulDesc_t   operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;

  // only support TN for FP8
  cublasOperation_t  transa  = CUBLAS_OP_T;
  cublasOperation_t  transb  = CUBLAS_OP_N;
  static const float h_alpha = 1.0;
  static const float h_beta  = 0.0;  // Can be non-zero starting from 12.0

  cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
  if (fastAccum) {
    const int8_t fastAccuMode = 1;  // enable fast imprecise accum
    cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(decltype(fastAccuMode)));
  }
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale));
  cudaDataType_t Atype = CUDA_R_8F_E4M3;
  cudaDataType_t Btype = CUDA_R_8F_E4M3;
  cudaDataType_t Ctype = CUDA_R_32F;
  cublasLtMatrixLayoutCreate(&Adesc, Atype, k, n, k);
  cublasLtMatrixLayoutCreate(&Bdesc, Btype, k, m, k);
  cublasLtMatrixLayoutCreate(&Cdesc, Ctype, n, m, n);
  cublasLtMatrixLayoutCreate(&Ddesc, Ctype, n, m, n);

  void* workSpace     = nullptr;
  int   workspaceSize = 0;

  auto ret = cublasLtMatmul(cublaslt_handle,
                            operationDesc,
                            &h_alpha,
                            A,
                            Adesc,
                            B,
                            Bdesc,
                            &h_beta,
                            nullptr,
                            Cdesc,
                            C,
                            Ddesc,
                            nullptr,
                            workSpace,
                            workspaceSize,
                            nullptr);
  if (ret) {
    throw std::runtime_error("cublasLtMatmul, " + std::to_string(ret));
  }

  CHECK_CUDA_ERROR();

  cublasLtMatmulDescDestroy(operationDesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Ddesc);

  CHECK_CUDA_ERROR();
}

// 错误检查宏
#define CHECK(call)                                                                                                    \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "Error: %s in file %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);                 \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CHECK_CUBLAS(call)                                                                                             \
  do {                                                                                                                 \
    cublasStatus_t status = call;                                                                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                                             \
      fprintf(stderr, "cuBLAS error: %d in file %s, line %d\n", status, __FILE__, __LINE__);                           \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

// 生成随机FP32矩阵
void generate_random_matrix(float* matrix, int rows, int cols)
{
  for (int i = 0; i < rows * cols; ++i) {
    matrix[i] = 1;
  }
}

// 计算per-tensor量化的缩放因子
float compute_scale_factor(const float* matrix, int size)
{
  float max_val = 0.0f;
  for (int i = 0; i < size; ++i) {
    max_val = std::max(max_val, std::abs(matrix[i]));
  }

  // FP8 E4M3的最大表示值约为448.0
  printf("compute_scale_factor, max_val = %8.5f, scale= %8.5f\n", max_val, max_val / 448.0f);
  return max_val / 448.0f;
}

// 将FP32矩阵量化为FP8 E4M3 (per-tensor)
void quantize_matrix_e4m3(const float* fp32_matrix, uint8_t* fp8_matrix, int rows, int cols, float scale)
{
  int size = rows * cols;
  for (int i = 0; i < size; ++i) {
    float         scaled_val = fp32_matrix[i] / scale;
    __nv_fp8_e4m3 tmp(scaled_val);
    fp8_matrix[i] = *(uint8_t*)&tmp;
  }
}

// 计算两个矩阵之间的最大绝对误差和均方误差
void calculate_errors(const float* fp32_result, const float* fp8_result, int size, float& max_error, float& mse)
{
  max_error        = 0.0f;
  float fp32_value = 0.0f;
  float fp8_value  = 0.0f;
  mse              = 0.0f;

  for (int i = 0; i < size; ++i) {
    float diff = fp32_result[i] - fp8_result[i];
    if (std::abs(diff) > max_error) {
      max_error  = std::abs(diff);
      fp32_value = fp32_result[i];
      fp8_value  = fp8_result[i];
    }
    mse += diff * diff;
  }

  printf("calculate_errors, max_error = %10.8f, max_relative_error = %10.8f, fp32 = %10.8f, fp8 = %10.8f\n",
         max_error,
         max_error / std::abs(fp32_value),
         fp32_value,
         fp8_value);

  mse /= size;
}

int main()
{
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;

  // 检查设备是否支持FP8
  int device;
  CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, device));

  std::cout << "使用设备: " << prop.name << std::endl;
  if (prop.major < 8) {
    std::cerr << "错误: 设备不支持FP8计算，需要Ampere或更新架构的GPU" << std::endl;
    return EXIT_FAILURE;
  }

  // 分配主机内存
  float* h_A             = new float[M * K];
  float* h_B             = new float[K * N];
  float* h_C_fp32        = new float[M * N];
  float* h_C_fp8_dequant = new float[M * N];

  uint8_t* h_A_fp8 = new uint8_t[M * K];
  uint8_t* h_B_fp8 = new uint8_t[K * N];

  // 生成随机矩阵
  std::cout << "生成随机矩阵..." << std::endl;
  generate_random_matrix(h_A, M, K);
  generate_random_matrix(h_B, K, N);

  // 计算per-tensor量化的缩放因子
  std::cout << "计算量化缩放因子..." << std::endl;
  float scale_A      = compute_scale_factor(h_A, M * K);
  float scale_B      = compute_scale_factor(h_B, K * N);
  float scale_output = scale_A * scale_B;  // 输出缩放因子

  // 量化矩阵到FP8 E4M3
  std::cout << "量化矩阵到FP8 E4M3..." << std::endl;
  quantize_matrix_e4m3(h_A, h_A_fp8, M, K, scale_A);
  quantize_matrix_e4m3(h_B, h_B_fp8, K, N, scale_B);

  // 分配设备内存
  float *  d_A, *d_B, *d_C_fp32;
  uint8_t *d_A_fp8, *d_B_fp8;
  float*   d_C_fp8;  // 用于存储FP8计算的FP32结果
  float *  d_A_scale, *d_B_scale;

  CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CHECK(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));
  CHECK(cudaMalloc(&d_A_fp8, M * K * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_B_fp8, K * N * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_C_fp8, M * N * sizeof(float)));
  CHECK(cudaMalloc(&d_A_scale, sizeof(float)));
  CHECK(cudaMalloc(&d_B_scale, sizeof(float)));

  // 复制数据到设备
  CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_A_fp8, h_A_fp8, M * K * sizeof(uint8_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B_fp8, h_B_fp8, K * N * sizeof(uint8_t), cudaMemcpyHostToDevice));

  // 初始化cuBLAS
  cublasHandle_t cublasH;
  CHECK_CUBLAS(cublasCreate(&cublasH));

  // 初始化cuBLASLt
  cublasLtHandle_t cublasLtH;
  CHECK_CUBLAS(cublasLtCreate(&cublasLtH));

  // 执行FP32矩阵乘法 (C = A * B)
  std::cout << "执行FP32矩阵乘法..." << std::endl;
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_fp32, N));

  CHECK(cudaMemcpy(d_A_scale, &scale_A, sizeof(float), cudaMemcpyDefault));
  CHECK(cudaMemcpy(d_B_scale, &scale_B, sizeof(float), cudaMemcpyDefault));
  // 执行FP8矩阵乘法
  std::cout << "执行FP8 E4M3矩阵乘法..." << std::endl;
  // fp8Gemm(const int m, const int n, const int k, const void *A, const int lda, const void *B, const int ldb, void *C,
  // const int ldc, float *a_scale, float *b_scale, bool fastAccum);
  fp8Gemm(M, N, K, d_A_fp8, K, d_B_fp8, N, d_C_fp8, N, d_A_scale, d_B_scale, true);

  // 复制结果回主机
  CHECK(cudaMemcpy(h_C_fp32, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_C_fp8_dequant, d_C_fp8, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // 计算误差
  float max_error, mse;
  calculate_errors(h_C_fp32, h_C_fp8_dequant, M * N, max_error, mse);

  std::cout << "精度比较结果:" << std::endl;
  std::cout << "最大绝对误差: " << max_error << std::endl;
  std::cout << "均方误差 (MSE): " << mse << std::endl;
  std::cout << "均方根误差 (RMSE): " << std::sqrt(mse) << std::endl;

  // 清理资源
  delete[] h_A;
  delete[] h_B;
  delete[] h_C_fp32;
  delete[] h_C_fp8_dequant;
  delete[] h_A_fp8;
  delete[] h_B_fp8;

  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C_fp32));
  CHECK(cudaFree(d_A_fp8));
  CHECK(cudaFree(d_B_fp8));
  CHECK(cudaFree(d_C_fp8));
  CHECK(cudaFree(d_A_scale));
  CHECK(cudaFree(d_B_scale));

  CHECK_CUBLAS(cublasLtDestroy(cublasLtH));
  CHECK_CUBLAS(cublasDestroy(cublasH));

  return EXIT_SUCCESS;
}
