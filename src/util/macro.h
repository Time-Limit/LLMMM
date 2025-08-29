#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR()                                                                                             \
  do {                                                                                                                 \
    cudaDeviceSynchronize();                                                                                           \
    cudaError_t err = cudaGetLastError();                                                                              \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                       \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CHECK_CUDA_ERROR_WITH_INFO(prefix)                                                                             \
  do {                                                                                                                 \
    cudaDeviceSynchronize();                                                                                           \
    cudaError_t err = cudaGetLastError();                                                                              \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "%s, CUDA error: %s at %s:%d\n", prefix, cudaGetErrorString(err), __FILE__, __LINE__);           \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CHECK_CUDA_RETURN(command)                                                                                     \
  do {                                                                                                                 \
    auto err = command;                                                                                                \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                       \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define CHECK_CUDA_RETURN_WITH_INFO(command, prefix)                                                                   \
  do {                                                                                                                 \
    auto err = command;                                                                                                \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "%s, CUDA error: %s at %s:%d\n", prefix, cudaGetErrorString(err), __FILE__, __LINE__);           \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define OFFSET(row, col, stride) ((row) * (stride) + (col))

#define FETCH_FLOAT(dst, src) *(float*)(&(dst)) = *(const float*)(&(src))

#define FETCH_FLOAT2(dst, src) *(float2*)(&(dst)) = *(const float2*)(&(src))

#define FETCH_FLOAT2_WITH_SRC_PTR(dst, src) *(float2*)(&(dst)) = *(const float2*)(src)

#define FETCH_FLOAT4(dst, src) *(float4*)(&(dst)) = *(const float4*)(&(src))

#define FETCH_FLOAT4_WITH_PTR(dst, src) *(float4*)(dst) = *(const float4*)(src)

#define FETCH_FLOAT4_WITH_SRC_PTR(dst, src) *(float4*)(&dst) = *(const float4*)(src)

#define FETCH_FLOAT4_PREFETCH_256B_WITH_SRC_PTR(dst, src)                                                              \
  {                                                                                                                    \
    asm volatile("ld.global.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];"                                                   \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_PREFETCH_64B_WITH_SRC_PTR(dst, src)                                                              \
  {                                                                                                                    \
    asm volatile("ld.global.L2::64B.v4.f32 {%0, %1, %2, %3}, [%4];"                                                   \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_PREFETCH_128B_WITH_SRC_PTR(dst, src)                                                              \
  {                                                                                                                    \
    asm volatile("ld.global.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"                                                   \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_EVICT_LAST_AND_PREFETCH_256B_WITH_SRC_PTR(dst, src)                                               \
  {                                                                                                                    \
    asm volatile("ld.global.L1::evict_last.L2::256B.v4.f32 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_CONST_PREFETCH_256B_WITH_SRC_PTR(dst, src)                                                        \
  {                                                                                                                    \
    asm volatile("ld.global.nc.L2::128B.v4.f32 {%0, %1, %2, %3}, [%4];"                                                \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define FETCH_FLOAT4_CONST_EVICT_LAST_WITH_SRC_PTR(dst, src)                                                           \
  {                                                                                                                    \
    asm volatile("ld.global.nc.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"                                          \
                 : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])                                              \
                 : "l"((const float*)(src)));                                                                          \
  }

#define STORE_FLOAT(dst, src) *(float*)(&(dst)) = *(const float*)(&(src))

#define STORE_FLOAT_WITH_PTR(dst, src) *(float*)((dst)) = *(const float*)((src))

#define STORE_FLOAT2(dst, src) *(float2*)(&(dst)) = *(const float2*)(&(src))

#define STORE_FLOAT4(dst, src) *(float4*)(&(dst)) = *(const float4*)(&(src))

#define STORE_FLOAT4_WITH_PTR(dst, src) *(float4*)(dst) = *(const float4*)(src)
