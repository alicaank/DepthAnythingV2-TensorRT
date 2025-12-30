# Example applications for Depth Anything V2 TensorRT

# Image inference example
add_executable(image_inference image_inference.cpp)
target_link_libraries(image_inference
    depth_anything_v2_trt
    ${OpenCV_LIBS}
)

# Video inference example
add_executable(video_inference video_inference.cpp)
target_link_libraries(video_inference
    depth_anything_v2_trt
    ${OpenCV_LIBS}
)

# Benchmark tool
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark
    depth_anything_v2_trt
    ${OpenCV_LIBS}
)

# Try to find NVML for power monitoring (optional)
find_library(NVML_LIBRARY nvml
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs
)

if(NVML_LIBRARY)
    target_link_libraries(benchmark ${NVML_LIBRARY})
    target_compile_definitions(benchmark PRIVATE HAVE_NVML)
    message(STATUS "NVML found: ${NVML_LIBRARY}")
else()
    message(STATUS "NVML not found - power monitoring will be disabled in benchmark")
endif()

# Install examples
install(TARGETS image_inference video_inference benchmark
    RUNTIME DESTINATION bin
)
