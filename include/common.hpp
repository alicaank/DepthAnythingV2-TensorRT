/**
 * Common types and constants for Depth Anything V2 TensorRT
 */

#ifndef DEPTH_ANYTHING_V2_COMMON_HPP
#define DEPTH_ANYTHING_V2_COMMON_HPP

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>

namespace depth_anything_v2 {

/**
 * Image normalization parameters (ImageNet statistics)
 * These MUST match the PyTorch preprocessing in depth_anything_v2/util/transform.py
 */
struct NormalizationParams {
    // RGB mean values
    float mean[3] = {0.485f, 0.456f, 0.406f};

    // RGB standard deviation values
    float std[3] = {0.229f, 0.224f, 0.225f};
};

/**
 * Model configuration
 */
struct ModelConfig {
    std::string encoder;      // Model encoder type (vits, vitb, vitl, vitg)
    int input_width;          // Input image width
    int input_height;         // Input image height
    int input_channels;       // Input image channels (always 3 for RGB)
    int output_width;         // Output depth map width
    int output_height;        // Output depth map height
};

/**
 * Inference configuration
 */
struct InferenceConfig {
    int device_id;            // CUDA device ID
    bool enable_cuda_graph;   // Enable CUDA Graph optimization
    bool use_pinned_memory;   // Use pinned host memory for faster transfers
    int num_streams;          // Number of CUDA streams for async execution
};

/**
 * Error handling macro
 */
#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t status = call;                                             \
        if (status != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error at ") +           \
                                     __FILE__ + ":" + std::to_string(__LINE__) + \
                                     " - " + cudaGetErrorString(status));      \
        }                                                                      \
    } while (0)

/**
 * CUDA kernel launch error checking
 */
#define CHECK_LAST_CUDA_ERROR()                                               \
    do {                                                                       \
        cudaError_t status = cudaGetLastError();                               \
        if (status != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA kernel error at ") +    \
                                     __FILE__ + ":" + std::to_string(__LINE__) + \
                                     " - " + cudaGetErrorString(status));      \
        }                                                                      \
    } while (0)

/**
 * RAII wrapper for CUDA device memory
 */
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    explicit DeviceMemory(size_t count) : size_(count * sizeof(T)) {
        CHECK_CUDA(cudaMalloc(&ptr_, size_));
    }

    ~DeviceMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Disable copy
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Enable move
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void* get() const { return ptr_; }
    size_t size() const { return size_; }

    void copyFrom(const void* host_ptr, size_t count, cudaStream_t stream = nullptr) {
        size_t bytes = count * sizeof(T);
        if (bytes > size_) {
            throw std::runtime_error("Copy size exceeds allocated memory");
        }
        if (stream) {
            CHECK_CUDA(cudaMemcpyAsync(ptr_, host_ptr, bytes, cudaMemcpyHostToDevice, stream));
        } else {
            CHECK_CUDA(cudaMemcpy(ptr_, host_ptr, bytes, cudaMemcpyHostToDevice));
        }
    }

    void copyTo(void* host_ptr, size_t count, cudaStream_t stream = nullptr) {
        size_t bytes = count * sizeof(T);
        if (bytes > size_) {
            throw std::runtime_error("Copy size exceeds allocated memory");
        }
        if (stream) {
            CHECK_CUDA(cudaMemcpyAsync(host_ptr, ptr_, bytes, cudaMemcpyDeviceToHost, stream));
        } else {
            CHECK_CUDA(cudaMemcpy(host_ptr, ptr_, bytes, cudaMemcpyDeviceToHost));
        }
    }

private:
    void* ptr_;
    size_t size_;
};

/**
 * RAII wrapper for CUDA pinned host memory
 */
template<typename T>
class PinnedMemory {
public:
    PinnedMemory() : ptr_(nullptr), size_(0) {}

    explicit PinnedMemory(size_t count) : size_(count * sizeof(T)) {
        CHECK_CUDA(cudaMallocHost(&ptr_, size_));
    }

    ~PinnedMemory() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }

    // Disable copy
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    // Enable move
    PinnedMemory(PinnedMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFreeHost(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() const { return static_cast<T*>(ptr_); }
    size_t size() const { return size_; }

private:
    void* ptr_;
    size_t size_;
};

/**
 * RAII wrapper for CUDA stream
 */
class CudaStream {
public:
    CudaStream() {
        CHECK_CUDA(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    // Disable copy
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Enable move
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }

    void synchronize() {
        CHECK_CUDA(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_;
};

/**
 * RAII wrapper for CUDA event
 */
class CudaEvent {
public:
    CudaEvent() {
        CHECK_CUDA(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    // Disable copy
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Enable move
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    cudaEvent_t get() const { return event_; }

    void record(cudaStream_t stream = nullptr) {
        CHECK_CUDA(cudaEventRecord(event_, stream));
    }

    void synchronize() {
        CHECK_CUDA(cudaEventSynchronize(event_));
    }

    float elapsedTime(const CudaEvent& start) const {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start.get(), event_));
        return ms;
    }

private:
    cudaEvent_t event_;
};

} // namespace depth_anything_v2

#endif // DEPTH_ANYTHING_V2_COMMON_HPP
