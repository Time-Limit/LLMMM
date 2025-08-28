#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace LLMMM {

template<typename T>
void transpose_matrix__aligned_128(T* dst, T* src, int M, int N, cudaStream_t stream);

void construct_m16n8k32_A_layout(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int M, int K, cudaStream_t stream);

void construct_m16n8k32_B_layout(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src_tranposed, int N, int K, cudaStream_t stream);

}
