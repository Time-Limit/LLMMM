find_package(CUDA REQUIRED)

add_library(fp16_mm STATIC mm.cu)

target_link_libraries(fp16_mm PRIVATE ${CUDA_LIBRARIES})
