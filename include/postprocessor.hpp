/**
 * Postprocessor for Depth Anything V2
 * Wraps GPU postprocessing kernel for efficient depth map upsampling
 */

#ifndef DEPTH_ANYTHING_V2_POSTPROCESSOR_HPP
#define DEPTH_ANYTHING_V2_POSTPROCESSOR_HPP

#include "common.hpp"
#include "cuda_utils.hpp"
#include <opencv2/opencv.hpp>

namespace depth_anything_v2 {

/**
 * GPU-accelerated depth map postprocessor
 *
 * Performs bilinear upsampling from model output to original image size
 * with align_corners=True (matching PyTorch's F.interpolate)
 */
class Postprocessor {
public:
    /**
     * Constructor
     *
     * @param model_output_width Model output width (typically 518)
     * @param model_output_height Model output height (typically 518)
     * @param device_id CUDA device ID
     */
    Postprocessor(
        int model_output_width = 518,
        int model_output_height = 518,
        int device_id = 0
    );

    /**
     * Destructor
     */
    ~Postprocessor();

    /**
     * Upsample depth map to target size (synchronous)
     *
     * @param d_input Input depth map (device pointer)
     * @param target_width Target width
     * @param target_height Target height
     * @param stream CUDA stream (nullptr for default stream)
     * @return Output depth map as cv::Mat (host memory)
     */
    cv::Mat upsample(
        const float* d_input,
        int target_width,
        int target_height,
        cudaStream_t stream = nullptr
    );

    /**
     * Upsample depth map to target size (asynchronous, device output)
     *
     * @param d_input Input depth map (device pointer)
     * @param d_output Output depth map (device pointer, must be pre-allocated)
     * @param target_width Target width
     * @param target_height Target height
     * @param stream CUDA stream
     */
    void upsampleAsync(
        const float* d_input,
        float* d_output,
        int target_width,
        int target_height,
        cudaStream_t stream = nullptr
    );

    /**
     * Normalize depth map to 0-255 range (for visualization)
     *
     * @param d_input Input depth map (device pointer)
     * @param width Depth map width
     * @param height Depth map height
     * @param stream CUDA stream (nullptr for default stream)
     * @return Normalized depth map as cv::Mat (8-bit grayscale)
     */
    cv::Mat normalizeForVisualization(
        const float* d_input,
        int width,
        int height,
        cudaStream_t stream = nullptr
    );

    /**
     * Get model output width
     */
    int getModelOutputWidth() const { return model_output_width_; }

    /**
     * Get model output height
     */
    int getModelOutputHeight() const { return model_output_height_; }

private:
    int model_output_width_;
    int model_output_height_;
    int device_id_;

    // Temporary device buffers
    DeviceMemory<float> d_output_buffer_;
    DeviceMemory<unsigned char> d_normalized_buffer_;
    size_t output_buffer_capacity_;
    size_t normalized_buffer_capacity_;

    void allocateOutputBuffer(size_t required_size);
    void allocateNormalizedBuffer(size_t required_size);

    // Helper to compute min/max for normalization
    void computeMinMax(const float* d_input, int size, float& min_val, float& max_val);
};

} // namespace depth_anything_v2

#endif // DEPTH_ANYTHING_V2_POSTPROCESSOR_HPP
