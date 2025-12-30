/**
 * Preprocessor Implementation
 */

#include "preprocessor.hpp"
#include <iostream>

namespace depth_anything_v2 {

Preprocessor::Preprocessor(int target_width, int target_height, int device_id)
    : target_width_(target_width),
      target_height_(target_height),
      device_id_(device_id),
      d_mean_(3),
      d_std_(3),
      input_buffer_capacity_(0)
{
    // Set CUDA device
    CHECK_CUDA(cudaSetDevice(device_id_));

    // Initialize normalization parameters (ImageNet statistics)
    // These MUST match PyTorch preprocessing: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    NormalizationParams params;

    // Copy to device
    d_mean_.copyFrom(params.mean, 3);
    d_std_.copyFrom(params.std, 3);
}

Preprocessor::~Preprocessor() {
    // RAII handles cleanup
}

void Preprocessor::allocateInputBuffer(size_t required_size) {
    if (required_size > input_buffer_capacity_) {
        // Allocate with some extra capacity to avoid frequent reallocations
        size_t new_capacity = required_size * 1.5;
        d_input_buffer_ = DeviceMemory<unsigned char>(new_capacity);
        input_buffer_capacity_ = new_capacity;
    }
}

void Preprocessor::process(const cv::Mat& input, float* d_output, cudaStream_t stream) {
    if (input.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    if (input.type() != CV_8UC3) {
        throw std::runtime_error("Input image must be CV_8UC3 (BGR)");
    }

    // Get input dimensions
    int input_width = input.cols;
    int input_height = input.rows;

    // Allocate input buffer if needed
    size_t input_size = input_width * input_height * 3;
    allocateInputBuffer(input_size);

    // Copy input to device
    d_input_buffer_.copyFrom(input.data, input_size, stream);

    // Launch preprocessing kernel
    kernels::launchPreprocessKernel(
        static_cast<unsigned char*>(d_input_buffer_.get()),
        d_output,
        input_width,
        input_height,
        target_width_,
        target_height_,
        static_cast<float*>(d_mean_.get()),
        static_cast<float*>(d_std_.get()),
        stream
    );

    CHECK_LAST_CUDA_ERROR();
}

void Preprocessor::processAsync(
    const unsigned char* d_input,
    int input_width,
    int input_height,
    float* d_output,
    cudaStream_t stream
) {
    // Launch preprocessing kernel directly on device memory
    kernels::launchPreprocessKernel(
        d_input,
        d_output,
        input_width,
        input_height,
        target_width_,
        target_height_,
        static_cast<float*>(d_mean_.get()),
        static_cast<float*>(d_std_.get()),
        stream
    );

    CHECK_LAST_CUDA_ERROR();
}

} // namespace depth_anything_v2
