/**
 * Postprocessor Implementation
 */

#include "postprocessor.hpp"
#include "cuda_utils.hpp"
#include <algorithm>
#include <iostream>

namespace depth_anything_v2 {

Postprocessor::Postprocessor(
    int model_output_width,
    int model_output_height,
    int device_id
)
    : model_output_width_(model_output_width),
      model_output_height_(model_output_height),
      device_id_(device_id),
      output_buffer_capacity_(0),
      normalized_buffer_capacity_(0)
{
    // Set CUDA device
    CHECK_CUDA(cudaSetDevice(device_id_));
}

Postprocessor::~Postprocessor() {
    // RAII handles cleanup
}

void Postprocessor::allocateOutputBuffer(size_t required_size) {
    if (required_size > output_buffer_capacity_) {
        // Allocate with some extra capacity
        size_t new_capacity = required_size * 1.5;
        d_output_buffer_ = DeviceMemory<float>(new_capacity / sizeof(float));
        output_buffer_capacity_ = new_capacity;
    }
}

void Postprocessor::allocateNormalizedBuffer(size_t required_size) {
    if (required_size > normalized_buffer_capacity_) {
        // Allocate with some extra capacity
        size_t new_capacity = required_size * 1.5;
        d_normalized_buffer_ = DeviceMemory<unsigned char>(new_capacity);
        normalized_buffer_capacity_ = new_capacity;
    }
}

void Postprocessor::computeMinMax(const float* d_input, int size, float& min_val, float& max_val) {
    // Use CUDA kernel to compute min/max on device
    kernels::computeMinMax(d_input, size, &min_val, &max_val, nullptr);
}

cv::Mat Postprocessor::upsample(
    const float* d_input,
    int target_width,
    int target_height,
    cudaStream_t stream
) {
    // Allocate output buffer
    size_t output_size = target_width * target_height * sizeof(float);
    allocateOutputBuffer(output_size);

    // Launch upsampling kernel
    kernels::launchUpsampleDepthKernel(
        d_input,
        static_cast<float*>(d_output_buffer_.get()),
        model_output_width_,
        model_output_height_,
        target_width,
        target_height,
        stream
    );

    CHECK_LAST_CUDA_ERROR();

    // Copy result to host
    std::vector<float> host_buffer(target_width * target_height);
    d_output_buffer_.copyTo(host_buffer.data(), host_buffer.size(), stream);

    if (stream) {
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Create cv::Mat
    cv::Mat result(target_height, target_width, CV_32FC1);
    std::memcpy(result.data, host_buffer.data(), output_size);

    return result;
}

void Postprocessor::upsampleAsync(
    const float* d_input,
    float* d_output,
    int target_width,
    int target_height,
    cudaStream_t stream
) {
    // Launch upsampling kernel directly to output buffer
    kernels::launchUpsampleDepthKernel(
        d_input,
        d_output,
        model_output_width_,
        model_output_height_,
        target_width,
        target_height,
        stream
    );

    CHECK_LAST_CUDA_ERROR();
}

cv::Mat Postprocessor::normalizeForVisualization(
    const float* d_input,
    int width,
    int height,
    cudaStream_t stream
) {
    int size = width * height;

    // Compute min/max
    float min_val, max_val;
    computeMinMax(d_input, size, min_val, max_val);

    // Allocate normalized buffer
    allocateNormalizedBuffer(size);

    // Launch normalization kernel
    kernels::launchNormalizeDepthKernel(
        d_input,
        static_cast<unsigned char*>(d_normalized_buffer_.get()),
        size,
        min_val,
        max_val,
        stream
    );

    CHECK_LAST_CUDA_ERROR();

    // Copy result to host
    std::vector<unsigned char> host_buffer(size);
    d_normalized_buffer_.copyTo(host_buffer.data(), size, stream);

    if (stream) {
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // Create cv::Mat
    cv::Mat result(height, width, CV_8UC1);
    std::memcpy(result.data, host_buffer.data(), size);

    return result;
}

} // namespace depth_anything_v2
