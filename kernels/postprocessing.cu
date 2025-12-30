/**
 * CUDA Postprocessing Kernel for Depth Anything V2
 *
 * Performs bilinear upsampling from model output (518×518) to original image size.
 * This MUST match PyTorch's F.interpolate(..., mode="bilinear", align_corners=True)
 * from depth_anything_v2/dpt.py lines 147, 192
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace depth_anything_v2 {
namespace kernels {

/**
 * Bilinear upsampling kernel with align_corners=True
 *
 * @param src_depth Input depth map [in_height, in_width]
 * @param dst_depth Output depth map [out_height, out_width]
 * @param in_width Input width
 * @param in_height Input height
 * @param out_width Output width
 * @param out_height Output height
 */
__global__ void upsampleDepthKernel(
    const float* __restrict__ src_depth,
    float* __restrict__ dst_depth,
    int in_width,
    int in_height,
    int out_width,
    int out_height
) {
    // Output pixel coordinates
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    // Calculate scale factors (align_corners=True mode)
    // This matches PyTorch's align_corners=True behavior
    float scale_x = (out_width > 1) ? static_cast<float>(in_width - 1) / (out_width - 1) : 0.0f;
    float scale_y = (out_height > 1) ? static_cast<float>(in_height - 1) / (out_height - 1) : 0.0f;

    // Map output coordinates to input coordinates
    float in_x = out_x * scale_x;
    float in_y = out_y * scale_y;

    // Get integer coordinates for bilinear interpolation
    int x0 = static_cast<int>(floorf(in_x));
    int y0 = static_cast<int>(floorf(in_y));
    int x1 = min(x0 + 1, in_width - 1);
    int y1 = min(y0 + 1, in_height - 1);

    // Clamp to valid range
    x0 = max(0, min(x0, in_width - 1));
    y0 = max(0, min(y0, in_height - 1));

    // Interpolation weights
    float wx = in_x - x0;
    float wy = in_y - y0;

    // Get four corner pixels
    float p00 = src_depth[y0 * in_width + x0];
    float p01 = src_depth[y0 * in_width + x1];
    float p10 = src_depth[y1 * in_width + x0];
    float p11 = src_depth[y1 * in_width + x1];

    // Bilinear interpolation
    float interpolated = (1.0f - wx) * (1.0f - wy) * p00 +
                         wx * (1.0f - wy) * p01 +
                         (1.0f - wx) * wy * p10 +
                         wx * wy * p11;

    // Store result
    dst_depth[out_y * out_width + out_x] = interpolated;
}

/**
 * Normalize depth map to 0-255 range for visualization
 *
 * @param src_depth Input depth map (float)
 * @param dst_depth Output normalized depth map (uint8)
 * @param size Number of pixels
 * @param min_val Minimum depth value
 * @param max_val Maximum depth value
 */
__global__ void normalizeDepthKernel(
    const float* __restrict__ src_depth,
    unsigned char* __restrict__ dst_depth,
    int size,
    float min_val,
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    float value = src_depth[idx];
    float range = max_val - min_val;

    // Avoid division by zero
    if (range < 1e-8f) {
        dst_depth[idx] = 0;
        return;
    }

    // Normalize to 0-255
    float normalized = (value - min_val) / range * 255.0f;
    normalized = fmaxf(0.0f, fminf(normalized, 255.0f));

    dst_depth[idx] = static_cast<unsigned char>(normalized);
}

/**
 * Launch upsampling kernel
 */
void launchUpsampleDepthKernel(
    const float* d_src_depth,
    float* d_dst_depth,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    cudaStream_t stream
) {
    // Launch configuration
    dim3 block(16, 16);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y
    );

    upsampleDepthKernel<<<grid, block, 0, stream>>>(
        d_src_depth,
        d_dst_depth,
        in_width,
        in_height,
        out_width,
        out_height
    );
}

/**
 * Launch normalization kernel
 */
void launchNormalizeDepthKernel(
    const float* d_src_depth,
    unsigned char* d_dst_depth,
    int size,
    float min_val,
    float max_val,
    cudaStream_t stream
) {
    // Launch configuration
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    normalizeDepthKernel<<<grid_size, block_size, 0, stream>>>(
        d_src_depth,
        d_dst_depth,
        size,
        min_val,
        max_val
    );
}

/**
 * Reduction kernel for finding min/max values
 */
__global__ void minMaxReductionKernel(
    const float* __restrict__ input,
    float* min_out,
    float* max_out,
    int size
) {
    extern __shared__ float shared_data[];
    float* shared_min = shared_data;
    float* shared_max = shared_data + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    float min_val = (idx < size) ? input[idx] : INFINITY;
    float max_val = (idx < size) ? input[idx] : -INFINITY;

    // Reduce within block
    shared_min[tid] = min_val;
    shared_max[tid] = max_val;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (idx + stride) < size) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        min_out[blockIdx.x] = shared_min[0];
        max_out[blockIdx.x] = shared_max[0];
    }
}

/**
 * Compute min and max values from device array
 */
void computeMinMax(
    const float* d_input,
    int size,
    float* min_val,
    float* max_val,
    cudaStream_t stream
) {
    // Launch configuration
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Allocate temporary buffers for reduction
    float* d_min_temp;
    float* d_max_temp;
    cudaMalloc(&d_min_temp, grid_size * sizeof(float));
    cudaMalloc(&d_max_temp, grid_size * sizeof(float));

    // First reduction pass
    size_t shared_mem_size = 2 * block_size * sizeof(float);
    minMaxReductionKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        d_input,
        d_min_temp,
        d_max_temp,
        size
    );

    // If we have more than one block, do another reduction
    while (grid_size > 1) {
        int new_size = grid_size;
        grid_size = (new_size + block_size - 1) / block_size;

        minMaxReductionKernel<<<grid_size, block_size, shared_mem_size, stream>>>(
            d_min_temp,
            d_min_temp,
            d_max_temp,
            new_size
        );
    }

    // Copy final results to host
    cudaMemcpyAsync(min_val, d_min_temp, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(max_val, d_max_temp, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Clean up
    cudaFree(d_min_temp);
    cudaFree(d_max_temp);
}

} // namespace kernels
} // namespace depth_anything_v2
