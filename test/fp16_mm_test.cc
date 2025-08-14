#include "fp16/mm.h"
#include "kahan.h"
#include "util/macro.h"
#include <random>

using namespace LLMMM;

template<typename T>
int test(int N, int K)
{
  const int                             M = 4096;
  std::vector<float>                    host_A(M * K), host_B(K * N), host_C(M * N);
  std::random_device                    rd;
  std::mt19937                          gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto& vec : {&host_A, &host_B}) {
    for (auto& data : *vec) {
      data = dis(gen);
    }
  }

  float *A, *B, *C;
  for (auto& pair : {std::make_pair(host_A, &A), std::make_pair(host_B, &B), std::make_pair(host_C, &C)}) {
    const std::vector<float>& host   = pair.first;
    float*&                   device = *pair.second;
    cudaMalloc(&device, sizeof(float) * host.size());
    cudaMemcpy(device, host.data(), sizeof(float) * host.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_fp32_naive_mm(A, B, C, M, N, K);
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  std::vector<T> host_fp16_A(M * K), host_fp16_B(K * N), host_fp16_C(M * N);
  for (auto [fp32, fp16] : {std::make_pair(&host_A, &host_fp16_A),
                            std::make_pair(&host_B, &host_fp16_B),
                            std::make_pair(&host_C, &host_fp16_C)}) {
    for (int i = 0; i < fp16->size(); ++i) {
      fp16->at(i) = T(fp32->at(i));
    }
  }

  T *fp16_A, *fp16_B, *fp16_C;
  for (auto& pair : {std::make_pair(host_fp16_A, &fp16_A),
                     std::make_pair(host_fp16_B, &fp16_B),
                     std::make_pair(host_fp16_C, &fp16_C)}) {
    const std::vector<T>& host   = pair.first;
    T*&                   device = *pair.second;
    cudaMalloc(&device, sizeof(T) * host.size());
    cudaMemcpy(device, host.data(), sizeof(T) * host.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }

  for (int i = 0; i < 8; ++i) {
    fp16_mm(fp16_A, fp16_B, fp16_C, M, N, K, nullptr);
    CHECK_CUDA_ERROR();
    cudaMemcpy(host_fp16_C.data(), fp16_C, sizeof(T) * host_fp16_C.size(), cudaMemcpyDefault);

    float max_error = 0, base_value, current_value;
    int   position  = 0;
    for (int i = 0; i < host_C.size(); ++i) {
      if (fabs(float(host_C[i]) - float(host_fp16_C[i])) > max_error) {
        max_error     = fabs(float(host_C[i]) - float(host_fp16_C[i]));
        base_value    = host_fp16_C[i];
        current_value = host_C[i];
        position      = i;
      }
    }
    const char* type = std::is_same<T, half>::value ? "half" : "__nv_bfloat16";
    const char* name = "fp16_mm";
    printf(
      "max_relative_error = %8.6f, max_absolute_error = %8.3f, base_value = %10.3f, current_value = %10.3f, type=%16s, function=%s\n",
      fabs(max_error / base_value),
      max_error,
      base_value,
      current_value,
      type,
      name);
  }

  CHECK_CUDA_RETURN(cudaFree(A));
  CHECK_CUDA_RETURN(cudaFree(B));
  CHECK_CUDA_RETURN(cudaFree(C));

  CHECK_CUDA_RETURN(cudaFree(fp16_A));
  CHECK_CUDA_RETURN(cudaFree(fp16_B));
  CHECK_CUDA_RETURN(cudaFree(fp16_C));

  return 0;
}

int main()
{
  test<half>(4096, 4096);
  // test<half>(6144, 2048);
  // test<half>(2048, 12288);
  // test<__nv_bfloat16>(4096, 4096);
  // test<__nv_bfloat16>(6144, 2048);
  // test<__nv_bfloat16>(2048, 12288);
  return 0;
}
