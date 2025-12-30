/**
 * Depth Anything V2 TensorRT Inference Engine Implementation
 */

#include "depth_anything_v2.hpp"
#include <fstream>
#include <iostream>
#include <cstring>

namespace depth_anything_v2 {

DepthAnythingV2::DepthAnythingV2(
    const std::string& engine_path,
    const InferenceConfig& config
)
    : runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      inference_config_(config),
      cuda_graph_enabled_(config.enable_cuda_graph),
      cuda_graph_captured_(false),
      cuda_graph_(nullptr),
      cuda_graph_exec_(nullptr)
{
    // Set CUDA device
    CHECK_CUDA(cudaSetDevice(config.device_id));

    // Load TensorRT engine
    loadEngine(engine_path);

    // Allocate GPU buffers
    allocateBuffers();

    // Create preprocessor and postprocessor
    preprocessor_ = std::make_unique<Preprocessor>(
        model_config_.input_width,
        model_config_.input_height,
        config.device_id
    );

    postprocessor_ = std::make_unique<Postprocessor>(
        model_config_.output_width,
        model_config_.output_height,
        config.device_id
    );

    // Initialize statistics
    stats_ = {0.0f, 0.0f, 0.0f, 0.0f, 0};

    // Warmup
    std::cout << "Warming up inference engine..." << std::endl;
    warmup(10);

    // Create CUDA Graph after warmup
    if (cuda_graph_enabled_) {
        std::cout << "Creating CUDA Graph for optimized inference..." << std::endl;
        createCudaGraph();
    }

    std::cout << "Depth Anything V2 TensorRT engine ready!" << std::endl;
}

DepthAnythingV2::~DepthAnythingV2() {
    // Destroy CUDA Graph
    if (cuda_graph_exec_) {
        cudaGraphExecDestroy(cuda_graph_exec_);
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
    }

    // Delete TensorRT objects (TensorRT 10+ uses delete instead of destroy)
    if (context_) {
        delete context_;
    }
    if (engine_) {
        delete engine_;
    }
    if (runtime_) {
        delete runtime_;
    }
}

void DepthAnythingV2::loadEngine(const std::string& engine_path) {
    std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;

    // Read engine file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    std::cout << "  Engine file size: " << size / 1024.0 / 1024.0 << " MB" << std::endl;

    // Deserialize engine
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    // Create execution context
    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }

    // Extract model configuration (TensorRT 10+ API)
    int num_io_tensors = engine_->getNbIOTensors();
    std::cout << "  Number of I/O tensors: " << num_io_tensors << std::endl;

    for (int i = 0; i < num_io_tensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);

        std::cout << "  Tensor " << i << ": " << name
                  << " (dims: [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
        std::cout << "]) - " << (is_input ? "INPUT" : "OUTPUT") << std::endl;

        if (is_input && std::string(name) == "image") {
            // Input tensor: [batch, channels, height, width]
            model_config_.input_channels = dims.d[1];
            model_config_.input_height = dims.d[2];
            model_config_.input_width = dims.d[3];
        } else if (!is_input && std::string(name) == "depth") {
            // Output tensor: [batch, height, width]
            model_config_.output_height = dims.d[1];
            model_config_.output_width = dims.d[2];
        }
    }

    model_config_.encoder = "unknown";  // Can't determine from engine

    std::cout << "  Model config:" << std::endl;
    std::cout << "    Input: " << model_config_.input_channels << "x"
              << model_config_.input_height << "x" << model_config_.input_width << std::endl;
    std::cout << "    Output: " << model_config_.output_height << "x"
              << model_config_.output_width << std::endl;
}

void DepthAnythingV2::allocateBuffers() {
    // Input buffer: [1, 3, H, W]
    size_t input_size = model_config_.input_channels *
                        model_config_.input_height *
                        model_config_.input_width;
    d_input_ = DeviceMemory<float>(input_size);

    // Output buffer: [1, H, W]
    size_t output_size = model_config_.output_height * model_config_.output_width;
    d_output_ = DeviceMemory<float>(output_size);

    std::cout << "Allocated GPU buffers:" << std::endl;
    std::cout << "  Input: " << input_size * sizeof(float) / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Output: " << output_size * sizeof(float) / 1024.0 / 1024.0 << " MB" << std::endl;
}

void DepthAnythingV2::createCudaGraph() {
    if (!cuda_graph_enabled_ || cuda_graph_captured_) {
        return;
    }

    try {
        // Set tensor addresses for TensorRT 10+
        context_->setTensorAddress("image", d_input_.get());
        context_->setTensorAddress("depth", d_output_.get());

        // Begin graph capture on dedicated stream
        CHECK_CUDA(cudaStreamBeginCapture(graph_stream_.get(), cudaStreamCaptureModeGlobal));

        // Execute inference once to capture the graph
        if (!context_->enqueueV3(graph_stream_.get())) {
            throw std::runtime_error("Failed to enqueue inference for CUDA Graph capture");
        }

        // End capture
        CHECK_CUDA(cudaStreamEndCapture(graph_stream_.get(), &cuda_graph_));

        // Instantiate the graph
        CHECK_CUDA(cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0));

        cuda_graph_captured_ = true;
        std::cout << "  ✓ CUDA Graph created successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ⚠ Warning: Failed to create CUDA Graph: " << e.what() << std::endl;
        std::cerr << "  Falling back to regular execution" << std::endl;
        cuda_graph_enabled_ = false;
    }
}

void DepthAnythingV2::executeCudaGraph(cudaStream_t stream) {
    if (cuda_graph_captured_) {
        CHECK_CUDA(cudaGraphLaunch(cuda_graph_exec_, stream ? stream : graph_stream_.get()));
    } else {
        throw std::runtime_error("CUDA Graph not captured");
    }
}

void DepthAnythingV2::executeInference(cudaStream_t stream) {
    // Set tensor addresses for TensorRT 10+
    context_->setTensorAddress("image", d_input_.get());
    context_->setTensorAddress("depth", d_output_.get());

    if (!context_->enqueueV3(stream)) {
        throw std::runtime_error("Failed to enqueue inference");
    }
}

cv::Mat DepthAnythingV2::infer(const cv::Mat& input) {
    if (input.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    // Get original dimensions
    int orig_height = input.rows;
    int orig_width = input.cols;

    // Create CUDA stream
    CudaStream stream;

    // 1. Preprocessing
    start_event_.record(stream.get());
    preprocessor_->process(input, static_cast<float*>(d_input_.get()), stream.get());
    end_event_.record(stream.get());
    end_event_.synchronize();
    recordPreprocessTime(end_event_.elapsedTime(start_event_));

    // 2. Inference
    start_event_.record(stream.get());
    if (cuda_graph_enabled_ && cuda_graph_captured_) {
        executeCudaGraph(stream.get());
    } else {
        executeInference(stream.get());
    }
    end_event_.record(stream.get());
    end_event_.synchronize();
    recordInferenceTime(end_event_.elapsedTime(start_event_));

    // 3. Postprocessing
    start_event_.record(stream.get());
    cv::Mat depth = postprocessor_->upsample(
        static_cast<float*>(d_output_.get()),
        orig_width,
        orig_height,
        stream.get()
    );
    end_event_.record(stream.get());
    end_event_.synchronize();
    recordPostprocessTime(end_event_.elapsedTime(start_event_));

    return depth;
}

void DepthAnythingV2::inferWithVisualization(
    const cv::Mat& input,
    cv::Mat& depth,
    cv::Mat& depth_vis
) {
    // Get depth map
    depth = infer(input);

    // Normalize for visualization
    double min_val, max_val;
    cv::minMaxLoc(depth, &min_val, &max_val);

    depth.convertTo(depth_vis, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
}

void DepthAnythingV2::inferAsync(
    const float* d_input,
    float* d_output,
    cudaStream_t stream
) {
    // Copy input to internal buffer
    size_t input_size = model_config_.input_channels *
                        model_config_.input_height *
                        model_config_.input_width;
    CHECK_CUDA(cudaMemcpyAsync(
        d_input_.get(),
        d_input,
        input_size * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream
    ));

    // Execute inference
    if (cuda_graph_enabled_ && cuda_graph_captured_) {
        executeCudaGraph(stream);
    } else {
        executeInference(stream);
    }

    // Copy output
    size_t output_size = model_config_.output_height * model_config_.output_width;
    CHECK_CUDA(cudaMemcpyAsync(
        d_output,
        d_output_.get(),
        output_size * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream
    ));
}

void DepthAnythingV2::warmup(int num_iterations) {
    // Create dummy input
    cv::Mat dummy_input(518, 518, CV_8UC3, cv::Scalar(128, 128, 128));

    for (int i = 0; i < num_iterations; ++i) {
        // Run inference without statistics
        CudaStream stream;
        preprocessor_->process(dummy_input, static_cast<float*>(d_input_.get()), stream.get());
        executeInference(stream.get());
        stream.synchronize();
    }

    std::cout << "  Warmup complete (" << num_iterations << " iterations)" << std::endl;
}

DepthAnythingV2::Statistics DepthAnythingV2::getStatistics() const {
    return stats_;
}

void DepthAnythingV2::resetStatistics() {
    stats_ = {0.0f, 0.0f, 0.0f, 0.0f, 0};
}

void DepthAnythingV2::recordPreprocessTime(float ms) const {
    float n = static_cast<float>(stats_.num_inferences);
    stats_.avg_preprocess_ms = (stats_.avg_preprocess_ms * n + ms) / (n + 1);
}

void DepthAnythingV2::recordInferenceTime(float ms) const {
    float n = static_cast<float>(stats_.num_inferences);
    stats_.avg_inference_ms = (stats_.avg_inference_ms * n + ms) / (n + 1);
}

void DepthAnythingV2::recordPostprocessTime(float ms) const {
    float n = static_cast<float>(stats_.num_inferences);
    stats_.avg_postprocess_ms = (stats_.avg_postprocess_ms * n + ms) / (n + 1);
    stats_.avg_total_ms = stats_.avg_preprocess_ms + stats_.avg_inference_ms + stats_.avg_postprocess_ms;
    stats_.num_inferences++;
}

} // namespace depth_anything_v2
