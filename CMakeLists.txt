cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(DepthAnythingV2TensorRT LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures (RTX 30xx-50xx)
# 75: Turing (RTX 20xx)
# 80: Ampere (A100)
# 86: Ampere (RTX 30xx)
# 89: Ada Lovelace (RTX 40xx, 50xx)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas=-v")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Find TensorRT
# TensorRT is typically installed in /usr/local/tensorrt or set via TensorRT_ROOT
if(DEFINED ENV{TensorRT_ROOT})
    set(TensorRT_ROOT $ENV{TensorRT_ROOT})
elseif(DEFINED TensorRT_ROOT)
    # Use the provided TensorRT_ROOT
else()
    set(TensorRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation directory")
endif()

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES include)

find_library(TensorRT_LIBRARY nvinfer
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TensorRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TensorRT_PARSERS_LIBRARY nvparsers
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TensorRT_ONNX_PARSER_LIBRARY nvonnxparser
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    set(TensorRT_FOUND TRUE)
    message(STATUS "Found TensorRT: ${TensorRT_LIBRARY}")
    message(STATUS "  TensorRT include: ${TensorRT_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "TensorRT not found. Please set TensorRT_ROOT or install TensorRT")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
)

# CUDA kernels library
add_subdirectory(kernels)

# Source files
set(SOURCES
    src/depth_anything_v2.cpp
    src/preprocessor.cpp
    src/postprocessor.cpp
    src/trt_logger.cpp
)

# Main library
add_library(depth_anything_v2_trt SHARED ${SOURCES})

target_link_libraries(depth_anything_v2_trt
    depth_kernels
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARY}
    ${TensorRT_PLUGIN_LIBRARY}
    ${TensorRT_ONNX_PARSER_LIBRARY}
    CUDA::cudart
)

# Set library properties
set_target_properties(depth_anything_v2_trt PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER "include/depth_anything_v2.hpp;include/preprocessor.hpp;include/postprocessor.hpp;include/common.hpp"
)

# Examples
add_subdirectory(examples)

# Installation
install(TARGETS depth_anything_v2_trt depth_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/depth_anything_v2
)

install(DIRECTORY include/
    DESTINATION include/depth_anything_v2
    FILES_MATCHING PATTERN "*.hpp"
)

# Print configuration summary
message(STATUS "")
message(STATUS "Depth Anything V2 TensorRT Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  TensorRT: ${TensorRT_LIBRARY}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "")
