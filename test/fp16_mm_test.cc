#include "mm.h"
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

  printf("start to fp32 mm ...\n");
  {
    cudaMemset(C, 0, M * N * sizeof(float));
    launch_fp32_mm(A, B, C, M, N, K);
    cudaMemcpy(host_C.data(), C, sizeof(float) * host_C.size(), cudaMemcpyDefault);
    CHECK_CUDA_ERROR();
  }
  printf("fp32 mm executed successfully.\n");

  LLMMM::MM<T> mm;
  mm.verify(host_A.data(), host_B.data(), host_C.data(), M, N, K);

  CHECK_CUDA_RETURN(cudaFree(A));
  CHECK_CUDA_RETURN(cudaFree(B));
  CHECK_CUDA_RETURN(cudaFree(C));
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
