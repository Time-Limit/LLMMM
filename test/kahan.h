#pragma once

namespace LLMMM {

void launch_kahan(const float* A, const float* B, float* C, int M, int N, int K);

void launch_fp32_mm(const float* A, const float* B, float* C, int M, int N, int K);

}  // namespace LLMMM
