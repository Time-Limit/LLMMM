add_library(fp16_mm_impl STATIC impl.cc mm.cu)

set_target_properties(fp16_mm_impl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
