/**
 * CUDA Preprocessing Kernel for Depth Anything V2
 *
 * Fused kernel that performs:
 * 1. Bilinear resize (arbitrary size → 518×518)
 * 2. BGR → RGB conversion
 * 3. Normalization: (pixel/255.0 - mean) / std
 * 4. HWC → CHW format conversion
 *
 * This MUST match the PyTorch preprocessing in:
 * - depth_anything_v2/util/transform.py (resize, normalize, transpose)
 * - depth_anything_v2/dpt.py (BGR→RGB, /255.0)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace depth_anything_v2 {
namespace kernels {

/**
 * Fused preprocessing kernel
 *
 * @param src_bgr Input BGR image in HWC format (uint8)
 * @param dst_rgb_chw Output RGB tensor in CHW format (float32)
 * @param src_width Input image width
 * @param src_height Input image height
 * @param dst_width Output width (typically 518)
 * @param dst_height Output height (typically 518)
 * @param mean RGB mean values [3]
 * @param std RGB standard deviation values [3]
 */
__global__ void preprocessKernel(
    const unsigned char* __restrict__ src_bgr,
    float* __restrict__ dst_rgb_chw,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    const float* __restrict__ mean,
    const float* __restrict__ std
) {
    // Output pixel coordinates
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= dst_width || out_y >= dst_height) {
        return;
    }

    // Calculate scale factors for bilinear interpolation
    float scale_x = static_cast<float>(src_width) / dst_width;
    float scale_y = static_cast<float>(src_height) / dst_height;

    // Map output coordinates to input coordinates (center-aligned)
    float in_x = (out_x + 0.5f) * scale_x - 0.5f;
    float in_y = (out_y + 0.5f) * scale_y - 0.5f;

    // Clamp to valid range
    in_x = fmaxf(0.0f, fminf(in_x, src_width - 1.0f));
    in_y = fmaxf(0.0f, fminf(in_y, src_height - 1.0f));

    // Get integer coordinates for bilinear interpolation
    int x0 = static_cast<int>(floorf(in_x));
    int y0 = static_cast<int>(floorf(in_y));
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);

    // Interpolation weights
    float wx = in_x - x0;
    float wy = in_y - y0;

    // Process all 3 channels
    for (int c = 0; c < 3; ++c) {
        // BGR → RGB conversion (reverse channel order)
        int src_c = 2 - c;  // BGR to RGB: 0→2, 1→1, 2→0

        // Get four corner pixels (HWC format: [H, W, C])
        int idx00 = (y0 * src_width + x0) * 3 + src_c;
        int idx01 = (y0 * src_width + x1) * 3 + src_c;
        int idx10 = (y1 * src_width + x0) * 3 + src_c;
        int idx11 = (y1 * src_width + x1) * 3 + src_c;

        float p00 = static_cast<float>(src_bgr[idx00]);
        float p01 = static_cast<float>(src_bgr[idx01]);
        float p10 = static_cast<float>(src_bgr[idx10]);
        float p11 = static_cast<float>(src_bgr[idx11]);

        // Bilinear interpolation
        float pixel_value = (1.0f - wx) * (1.0f - wy) * p00 +
                            wx * (1.0f - wy) * p01 +
                            (1.0f - wx) * wy * p10 +
                            wx * wy * p11;

        // Normalize: (pixel/255 - mean) / std
        // This matches PyTorch preprocessing exactly
        float normalized = (pixel_value / 255.0f - mean[c]) / std[c];

        // Store in CHW format: [C, H, W]
        int dst_idx = c * (dst_height * dst_width) + out_y * dst_width + out_x;
        dst_rgb_chw[dst_idx] = normalized;
    }
}

/**
 * Launch preprocessing kernel
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
    cudaStream_t stream
) {
    // Launch configuration
    dim3 block(16, 16);
    dim3 grid(
        (dst_width + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y
    );

    preprocessKernel<<<grid, block, 0, stream>>>(
        d_src_bgr,
        d_dst_rgb_chw,
        src_width,
        src_height,
        dst_width,
        dst_height,
        d_mean,
        d_std
    );
}

} // namespace kernels
} // namespace depth_anything_v2
