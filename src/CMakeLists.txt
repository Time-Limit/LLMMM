add_subdirectory(fp16)

add_library(llmmm STATIC mm.cc)

target_link_libraries(llmmm PRIVATE fp16_mm_impl)
