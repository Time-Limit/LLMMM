#include "util/macro.h"
#include "util/util.h"
#include <cuda_fp8.h>
#include <random>

using namespace LLMMM;

template<typename T>
int test(int M, int N)
{
  std::vector<T>                        host(M * N);
  std::random_device                    rd;
  std::mt19937                          gen(rd());
  std::uniform_real_distribution<float> dis(-5, 5);
  for (auto& data : host) {
    data = T(dis(gen));
  }

  T* device;
  CHECK_CUDA_RETURN(cudaMalloc(&device, sizeof(T) * host.size()));
  CHECK_CUDA_RETURN(cudaMemcpy(device, host.data(), sizeof(T) * host.size(), cudaMemcpyDefault));

  T* transposed_device;
  CHECK_CUDA_RETURN(cudaMalloc(&transposed_device, sizeof(T) * host.size()));

  LLMMM::transpose_matrix__aligned_128(transposed_device, device, M, N, nullptr);
  CHECK_CUDA_ERROR();
  std::vector<T> transposed_host(M * N);
  CHECK_CUDA_RETURN(
    cudaMemcpy(transposed_host.data(), transposed_device, sizeof(T) * transposed_host.size(), cudaMemcpyDefault));

  int error = 0, correct = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      if (memcmp(&host[OFFSET(m, n, N)], &transposed_host[OFFSET(n, m, M)], sizeof(T))) {
        error++;
      }
      else {
        correct++;
      }
    }
  }

  const char* type = typeid(T).name();
  printf("T = %16s, M = %5d, N = %5d, error = %5d, correct = %5d\n", type, M, N, error, correct);

  CHECK_CUDA_RETURN(cudaFree(device));
  CHECK_CUDA_RETURN(cudaFree(transposed_device));

  return 0;
}

int main()
{
  test<float>(4096, 4096);
  test<float>(6144, 2048);
  test<float>(2048, 12288);
  test<__nv_fp8_e4m3>(4096, 4096);
  test<__nv_fp8_e4m3>(6144, 2048);
  test<__nv_fp8_e4m3>(2048, 12288);
  return 0;
}
