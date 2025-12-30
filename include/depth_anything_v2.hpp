/**
 * Depth Anything V2 TensorRT Inference Engine
 */

#ifndef DEPTH_ANYTHING_V2_HPP
#define DEPTH_ANYTHING_V2_HPP

#include "common.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "trt_logger.hpp"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <vector>

namespace depth_anything_v2 {

/**
 * Main TensorRT inference engine for Depth Anything V2
 *
 * Features:
 * - FP16 precision support
 * - CUDA Graph optimization for reduced kernel launch overhead
 * - GPU preprocessing and postprocessing
 * - Async execution with CUDA streams
 */
class DepthAnythingV2 {
public:
    /**
     * Constructor
     *
     * @param engine_path Path to TensorRT engine file (.trt)
     * @param config Inference configuration
     */
    explicit DepthAnythingV2(
        const std::string& engine_path,
        const InferenceConfig& config = InferenceConfig{0, true, true, 1}
    );

    /**
     * Destructor
     */
    ~DepthAnythingV2();

    /**
     * Infer depth map from BGR image (synchronous)
     *
     * @param input Input BGR image (cv::Mat)
     * @return Depth map (same size as input image)
     */
    cv::Mat infer(const cv::Mat& input);

    /**
     * Infer depth map with visualization (synchronous)
     *
     * @param input Input BGR image
     * @param depth Output depth map (float, same size as input)
     * @param depth_vis Output normalized depth visualization (8-bit grayscale)
     */
    void inferWithVisualization(
        const cv::Mat& input,
        cv::Mat& depth,
        cv::Mat& depth_vis
    );

    /**
     * Async inference (for pipeline processing)
     *
     * @param d_input Preprocessed input on device (CHW format)
     * @param d_output Depth output on device (HW format, 518x518)
     * @param stream CUDA stream
     */
    void inferAsync(
        const float* d_input,
        float* d_output,
        cudaStream_t stream = nullptr
    );

    /**
     * Get model configuration
     */
    const ModelConfig& getModelConfig() const { return model_config_; }

    /**
     * Get inference statistics
     */
    struct Statistics {
        float avg_preprocess_ms;
        float avg_inference_ms;
        float avg_postprocess_ms;
        float avg_total_ms;
        size_t num_inferences;
    };

    Statistics getStatistics() const;
    void resetStatistics();

    /**
     * Warmup (run several dummy inferences to initialize CUDA Graph)
     *
     * @param num_iterations Number of warmup iterations (default: 10)
     */
    void warmup(int num_iterations = 10);

private:
    // TensorRT components
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    // Model configuration
    ModelConfig model_config_;
    InferenceConfig inference_config_;

    // GPU buffers for inference
    DeviceMemory<float> d_input_;
    DeviceMemory<float> d_output_;

    // CUDA Graph (for optimization)
    bool cuda_graph_enabled_;
    bool cuda_graph_captured_;
    cudaGraph_t cuda_graph_;
    cudaGraphExec_t cuda_graph_exec_;
    CudaStream graph_stream_;

    // Preprocessing and postprocessing
    std::unique_ptr<Preprocessor> preprocessor_;
    std::unique_ptr<Postprocessor> postprocessor_;

    // Performance tracking
    mutable Statistics stats_;
    mutable CudaEvent start_event_;
    mutable CudaEvent end_event_;

    // Private methods
    void loadEngine(const std::string& engine_path);
    void allocateBuffers();
    void createCudaGraph();
    void executeCudaGraph(cudaStream_t stream);
    void executeInference(cudaStream_t stream);

    // Timing helpers
    void recordPreprocessTime(float ms) const;
    void recordInferenceTime(float ms) const;
    void recordPostprocessTime(float ms) const;
};

} // namespace depth_anything_v2

#endif // DEPTH_ANYTHING_V2_HPP
