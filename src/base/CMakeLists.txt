find_package(CUDA REQUIRED)

add_library(base_kernel STATIC base.cu)

target_link_libraries(base_kernel PRIVATE ${CUDA_LIBRARIES})
