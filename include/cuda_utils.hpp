/**
 * CUDA Utilities and Kernel Declarations for Depth Anything V2
 */

#ifndef DEPTH_ANYTHING_V2_CUDA_UTILS_HPP
#define DEPTH_ANYTHING_V2_CUDA_UTILS_HPP

#include <cuda_runtime.h>

namespace depth_anything_v2 {
namespace kernels {

/**
 * Launch preprocessing kernel
 *
 * Fused kernel that performs resize, BGR→RGB, normalization, and HWC→CHW conversion
 *
 * @param d_src_bgr Input BGR image (uint8, HWC format)
 * @param d_dst_rgb_chw Output RGB tensor (float32, CHW format)
 * @param src_width Input image width
 * @param src_height Input image height
 * @param dst_width Output width (typically 518)
 * @param dst_height Output height (typically 518)
 * @param d_mean Device pointer to RGB mean values [3]
 * @param d_std Device pointer to RGB std values [3]
 * @param stream CUDA stream for async execution
 */
void launchPreprocessKernel(
    const unsigned char* d_src_bgr,
    float* d_dst_rgb_chw,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    const float* d_mean,
    const float* d_std,
    cudaStream_t stream = nullptr
);

/**
 * Launch depth upsampling kernel
 *
 * Bilinear upsampling with align_corners=True (matches PyTorch)
 *
 * @param d_src_depth Input depth map
 * @param d_dst_depth Output depth map
 * @param in_width Input width
 * @param in_height Input height
 * @param out_width Output width
 * @param out_height Output height
 * @param stream CUDA stream for async execution
 */
void launchUpsampleDepthKernel(
    const float* d_src_depth,
    float* d_dst_depth,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    cudaStream_t stream = nullptr
);

/**
 * Launch depth normalization kernel
 *
 * Normalize depth map to 0-255 range for visualization
 *
 * @param d_src_depth Input depth map (float)
 * @param d_dst_depth Output normalized depth (uint8)
 * @param size Number of pixels
 * @param min_val Minimum depth value
 * @param max_val Maximum depth value
 * @param stream CUDA stream for async execution
 */
void launchNormalizeDepthKernel(
    const float* d_src_depth,
    unsigned char* d_dst_depth,
    int size,
    float min_val,
    float max_val,
    cudaStream_t stream = nullptr
);

/**
 * Compute min and max values from device array
 *
 * @param d_input Input device array
 * @param size Number of elements
 * @param min_val Output minimum value
 * @param max_val Output maximum value
 * @param stream CUDA stream for async execution
 */
void computeMinMax(
    const float* d_input,
    int size,
    float* min_val,
    float* max_val,
    cudaStream_t stream = nullptr
);

} // namespace kernels
} // namespace depth_anything_v2

#endif // DEPTH_ANYTHING_V2_CUDA_UTILS_HPP
