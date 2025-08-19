# test/CMakeLists.txt

add_library(kahan OBJECT kahan.cu)

add_executable(fp16_mm_test fp16_mm_test.cc)
target_link_libraries(fp16_mm_test PRIVATE ${CUDA_LIBRARIES} fp16_mm kahan)

add_executable(transpose_matrix__aligned_128_test transpose_matrix__aligned_128_test.cc)
target_link_libraries(transpose_matrix__aligned_128_test PRIVATE ${CUDA_LIBRARIES} base_kernel)
