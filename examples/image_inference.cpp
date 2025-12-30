/**
 * Single Image Inference Example for Depth Anything V2
 *
 * Usage:
 *   ./image_inference --engine <path_to_engine> --input <input_image> --output <output_image>
 */

#include "depth_anything_v2.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace depth_anything_v2;

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --engine <path>     Path to TensorRT engine file (required)\n"
              << "  --input <path>      Input image path (required)\n"
              << "  --output <path>     Output depth image path (default: depth_output.png)\n"
              << "  --colormap          Apply colormap to depth visualization\n"
              << "  --device <id>       CUDA device ID (default: 0)\n"
              << "  --help              Show this help message\n"
              << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string engine_path;
    std::string input_path;
    std::string output_path = "depth_output.png";
    bool use_colormap = false;
    int device_id = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--colormap") {
            use_colormap = true;
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate arguments
    if (engine_path.empty() || input_path.empty()) {
        std::cerr << "Error: --engine and --input are required\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Depth Anything V2 - Image Inference" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Engine: " << engine_path << std::endl;
        std::cout << "Input: " << input_path << std::endl;
        std::cout << "Output: " << output_path << std::endl;
        std::cout << "Device: " << device_id << std::endl;
        std::cout << "Colormap: " << (use_colormap ? "Yes" : "No") << std::endl;
        std::cout << std::endl;

        // Load input image
        std::cout << "Loading input image..." << std::endl;
        cv::Mat input = cv::imread(input_path, cv::IMREAD_COLOR);
        if (input.empty()) {
            throw std::runtime_error("Failed to load input image: " + input_path);
        }
        std::cout << "  Image size: " << input.cols << "x" << input.rows << std::endl;
        std::cout << std::endl;

        // Initialize inference engine
        InferenceConfig config;
        config.device_id = device_id;
        config.enable_cuda_graph = true;
        config.use_pinned_memory = true;

        DepthAnythingV2 engine(engine_path, config);

        // Run inference
        std::cout << "Running inference..." << std::endl;
        cv::Mat depth, depth_vis;
        engine.inferWithVisualization(input, depth, depth_vis);
        std::cout << "  ✓ Inference complete" << std::endl;
        std::cout << std::endl;

        // Apply colormap if requested
        cv::Mat output;
        if (use_colormap) {
            cv::applyColorMap(depth_vis, output, cv::COLORMAP_INFERNO);
        } else {
            output = depth_vis;
        }

        // Save output
        std::cout << "Saving output..." << std::endl;
        cv::imwrite(output_path, output);
        std::cout << "  ✓ Saved to: " << output_path << std::endl;
        std::cout << std::endl;

        // Print statistics
        auto stats = engine.getStatistics();
        std::cout << "Performance Statistics:" << std::endl;
        std::cout << "  Preprocessing: " << stats.avg_preprocess_ms << " ms" << std::endl;
        std::cout << "  Inference:     " << stats.avg_inference_ms << " ms" << std::endl;
        std::cout << "  Postprocessing: " << stats.avg_postprocess_ms << " ms" << std::endl;
        std::cout << "  Total:         " << stats.avg_total_ms << " ms" << std::endl;
        std::cout << "  FPS:           " << (1000.0f / stats.avg_total_ms) << std::endl;
        std::cout << std::endl;

        std::cout << "✓ Complete!" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
