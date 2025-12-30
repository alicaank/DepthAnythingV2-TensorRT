/**
 * Preprocessor for Depth Anything V2
 * Wraps GPU preprocessing kernel for efficient image preprocessing
 */

#ifndef DEPTH_ANYTHING_V2_PREPROCESSOR_HPP
#define DEPTH_ANYTHING_V2_PREPROCESSOR_HPP

#include "common.hpp"
#include "cuda_utils.hpp"
#include <opencv2/opencv.hpp>

namespace depth_anything_v2 {

/**
 * GPU-accelerated image preprocessor
 *
 * Performs fused preprocessing on GPU:
 * 1. Resize to target size (bilinear interpolation)
 * 2. BGR → RGB conversion
 * 3. Normalization with ImageNet statistics
 * 4. HWC → CHW format conversion
 */
class Preprocessor {
public:
    /**
     * Constructor
     *
     * @param target_width Target width (typically 518)
     * @param target_height Target height (typically 518)
     * @param device_id CUDA device ID
     */
    Preprocessor(int target_width = 518, int target_height = 518, int device_id = 0);

    /**
     * Destructor
     */
    ~Preprocessor();

    /**
     * Preprocess image (synchronous)
     *
     * @param input Input BGR image (cv::Mat)
     * @param output Output buffer (device pointer, CHW format)
     * @param stream CUDA stream (nullptr for default stream)
     */
    void process(const cv::Mat& input, float* d_output, cudaStream_t stream = nullptr);

    /**
     * Preprocess image from device memory (async)
     *
     * @param d_input Input BGR image on device (HWC format)
     * @param input_width Input image width
     * @param input_height Input image height
     * @param d_output Output buffer (device pointer, CHW format)
     * @param stream CUDA stream
     */
    void processAsync(
        const unsigned char* d_input,
        int input_width,
        int input_height,
        float* d_output,
        cudaStream_t stream = nullptr
    );

    /**
     * Get target width
     */
    int getTargetWidth() const { return target_width_; }

    /**
     * Get target height
     */
    int getTargetHeight() const { return target_height_; }

    /**
     * Get output tensor size (in bytes)
     */
    size_t getOutputSize() const {
        return 3 * target_width_ * target_height_ * sizeof(float);
    }

private:
    int target_width_;
    int target_height_;
    int device_id_;

    // Normalization parameters (device memory)
    DeviceMemory<float> d_mean_;
    DeviceMemory<float> d_std_;

    // Temporary device buffer for input image
    DeviceMemory<unsigned char> d_input_buffer_;
    size_t input_buffer_capacity_;

    void allocateInputBuffer(size_t required_size);
};

} // namespace depth_anything_v2

#endif // DEPTH_ANYTHING_V2_PREPROCESSOR_HPP
