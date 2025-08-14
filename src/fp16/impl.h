#pragma once

#include "mm_impl.h"
#include "util/macro.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mutex>
#include <tuple>
#include <vector>

namespace LLMMM {

template<typename T>
class FP16_MM_IMPL {
  static_assert(std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>);
  struct Kernel {
    int BLOCK_M;
    int BLOCK_N;
    int BLOCK_M_SPLIT_COUNT;
    int BLOCK_N_SPLIT_COUNT;
    int WARP_M;
    int WARP_N;
    using launcher_type = void (*)(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream);
    launcher_type launcher;
  };

  static std::vector<Kernel> kernels;

  std::mutex kernels_init_mutex;

  void init_kernels();

public:
  FP16_MM_IMPL() {
    if (kernels.empty()) {
      std::unique_lock lock(kernels_init_mutex);
      if (kernels.empty()) {
        init_kernels();
      }
    }
  }
  void operator()(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream)
  {
    return;
  }

  void tune(int N, int K)
  {
    return;
  }

  void verify(const float* host_A, const float* host_B, const float* host_benchmark, int M, int N, int K)
  {
    printf("start to verify %ld kernels\n", kernels.size());
    std::vector<T> host_T_A(M * K), host_T_B(K * N), host_T_C(M * N);
    T *            device_A, *device_B, *device_C;
    for (auto& [host_F, host_T, device] : {std::make_tuple(host_A, &host_T_A, &device_A),
                                           std::make_tuple(host_B, &host_T_B, &device_B),
                                           std::make_tuple((const float*)(nullptr), &host_T_C, &device_C)}) {
      CHECK_CUDA_RETURN(cudaMalloc(device, host_T->size() * sizeof(T)));
      if (host_F) {
        for (int i = 0; i < host_T->size(); ++i) {
          host_T->at(i) = T(host_F[i]);
        }
        CHECK_CUDA_RETURN(
          cudaMemcpy(*device, host_T->data(), sizeof(T) * host_T->size(), cudaMemcpyKind::cudaMemcpyHostToDevice));
      }
    }
    for (const auto& kernel : kernels) {
      printf("start to mm ...\n");
      for (int i = 0; i < 8; ++i) {
        kernel.launcher(device_A, device_B, device_C, M, N, K, nullptr);
        CHECK_CUDA_ERROR();
        CHECK_CUDA_RETURN(
          cudaMemcpy(host_T_C.data(), device_C, sizeof(T) * host_T_C.size(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        float max_error = 0, base_value, current_value;
        int   position  = 0;
        for (int i = 0; i < host_T_C.size(); ++i) {
          if (fabs(float(host_T_C[i]) - host_benchmark[i]) > max_error) {
            max_error     = fabs(float(host_T_C[i]) - host_benchmark[i]);
            base_value    = host_benchmark[i];
            current_value = host_T_C[i];
            position      = i;
          }
        }
        const char* type = std::is_same<T, half>::value ? "half" : "__nv_bfloat16";
        printf(
          "max_relative_error = %8.6f, max_absolute_error = %8.3f, base_value = %10.3f, current_value = %10.3f, type=%16s\n",
          fabs(max_error / base_value),
          max_error,
          base_value,
          current_value,
          type);
      }
    }
    CHECK_CUDA_RETURN(cudaFree(device_A));
    CHECK_CUDA_RETURN(cudaFree(device_B));
    CHECK_CUDA_RETURN(cudaFree(device_C));
  }
};

template<typename T>
std::vector<typename FP16_MM_IMPL<T>::Kernel> FP16_MM_IMPL<T>::kernels;

template<>
class MM_IMPL<half>: public FP16_MM_IMPL<half> {};

template<>
class MM_IMPL<__nv_bfloat16>: public FP16_MM_IMPL<__nv_bfloat16> {};

}  // namespace LLMMM
