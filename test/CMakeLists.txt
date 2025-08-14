# test/CMakeLists.txt

add_library(kahan OBJECT kahan.cu)

add_executable(fp16_mm_test fp16_mm_test.cc)
target_link_libraries(fp16_mm_test PRIVATE ${CUDA_LIBRARIES} llmmm kahan)
