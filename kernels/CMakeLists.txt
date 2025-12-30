# CUDA Kernels for Depth Anything V2 TensorRT

# CUDA source files
set(CUDA_SOURCES
    preprocessing.cu
    postprocessing.cu
)

# Create CUDA kernels library
add_library(depth_kernels STATIC ${CUDA_SOURCES})

# Set CUDA properties
set_target_properties(depth_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# CUDA compile options
target_compile_options(depth_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Xptxas=-v
        --generate-line-info
    >
)

# Include directories
target_include_directories(depth_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Link CUDA runtime
target_link_libraries(depth_kernels
    ${CUDA_LIBRARIES}
    cudart
)
