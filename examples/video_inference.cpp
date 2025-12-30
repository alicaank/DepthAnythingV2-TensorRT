/**
 * Video Inference Example for Depth Anything V2
 *
 * Features:
 * - Real-time video processing
 * - Side-by-side visualization
 * - Performance metrics
 *
 * Usage:
 *   ./video_inference --engine <path_to_engine> --input <video_file> [--output <output_file>]
 */

#include "depth_anything_v2.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

using namespace depth_anything_v2;

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --engine <path>     Path to TensorRT engine file (required)\n"
              << "  --input <path>      Input video file or camera index (required)\n"
              << "  --output <path>     Output video file (optional)\n"
              << "  --colormap          Apply colormap to depth visualization\n"
              << "  --display           Show live preview window\n"
              << "  --device <id>       CUDA device ID (default: 0)\n"
              << "  --help              Show this help message\n"
              << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string engine_path;
    std::string input_path;
    std::string output_path;
    bool use_colormap = false;
    bool display = false;
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
        } else if (arg == "--display") {
            display = true;
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
        std::cout << "Depth Anything V2 - Video Inference" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Engine: " << engine_path << std::endl;
        std::cout << "Input: " << input_path << std::endl;
        if (!output_path.empty()) {
            std::cout << "Output: " << output_path << std::endl;
        }
        std::cout << "Device: " << device_id << std::endl;
        std::cout << "Colormap: " << (use_colormap ? "Yes" : "No") << std::endl;
        std::cout << "Display: " << (display ? "Yes" : "No") << std::endl;
        std::cout << std::endl;

        // Open video
        std::cout << "Opening video..." << std::endl;
        cv::VideoCapture cap;

        // Try to parse as camera index
        try {
            int camera_idx = std::stoi(input_path);
            cap.open(camera_idx);
        } catch (...) {
            // Not a number, treat as file path
            cap.open(input_path);
        }

        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video: " + input_path);
        }

        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "  Resolution: " << frame_width << "x" << frame_height << std::endl;
        std::cout << "  FPS: " << fps << std::endl;
        std::cout << "  Total frames: " << total_frames << std::endl;
        std::cout << std::endl;

        // Initialize video writer
        cv::VideoWriter writer;
        if (!output_path.empty()) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            // Side-by-side output (original + depth)
            writer.open(output_path, fourcc, fps, cv::Size(frame_width * 2, frame_height));
            if (!writer.isOpened()) {
                throw std::runtime_error("Failed to create output video: " + output_path);
            }
        }

        // Initialize inference engine
        InferenceConfig config;
        config.device_id = device_id;
        config.enable_cuda_graph = true;
        config.use_pinned_memory = true;

        DepthAnythingV2 engine(engine_path, config);

        // Create display window
        if (display) {
            cv::namedWindow("Depth Anything V2", cv::WINDOW_NORMAL);
        }

        // Process video
        std::cout << "Processing video..." << std::endl;
        cv::Mat frame, depth, depth_vis;
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            // Read frame
            if (!cap.read(frame)) {
                break;
            }

            frame_count++;

            // Run inference
            engine.inferWithVisualization(frame, depth, depth_vis);

            // Apply colormap if requested
            cv::Mat depth_colored;
            if (use_colormap) {
                cv::applyColorMap(depth_vis, depth_colored, cv::COLORMAP_INFERNO);
            } else {
                cv::cvtColor(depth_vis, depth_colored, cv::COLOR_GRAY2BGR);
            }

            // Create side-by-side visualization
            cv::Mat combined;
            cv::hconcat(frame, depth_colored, combined);

            // Write to output video
            if (writer.isOpened()) {
                writer.write(combined);
            }

            // Display
            if (display) {
                cv::imshow("Depth Anything V2", combined);
                if (cv::waitKey(1) == 27) {  // ESC to quit
                    break;
                }
            }

            // Print progress
            if (frame_count % 30 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time
                ).count();
                float avg_fps = frame_count * 1000.0f / elapsed;

                std::cout << "\rProcessed " << frame_count << "/" << total_frames
                          << " frames (avg FPS: " << std::fixed << std::setprecision(1)
                          << avg_fps << ")" << std::flush;
            }
        }

        std::cout << std::endl << std::endl;

        // Print final statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();

        float avg_fps = frame_count * 1000.0f / total_time;

        auto stats = engine.getStatistics();

        std::cout << "Processing Complete!" << std::endl;
        std::cout << "  Total frames: " << frame_count << std::endl;
        std::cout << "  Total time: " << total_time / 1000.0f << " seconds" << std::endl;
        std::cout << "  Average FPS: " << avg_fps << std::endl;
        std::cout << std::endl;

        std::cout << "Per-Frame Statistics:" << std::endl;
        std::cout << "  Preprocessing:  " << stats.avg_preprocess_ms << " ms" << std::endl;
        std::cout << "  Inference:      " << stats.avg_inference_ms << " ms" << std::endl;
        std::cout << "  Postprocessing: " << stats.avg_postprocess_ms << " ms" << std::endl;
        std::cout << "  Total:          " << stats.avg_total_ms << " ms" << std::endl;
        std::cout << std::endl;

        if (!output_path.empty()) {
            std::cout << "✓ Output saved to: " << output_path << std::endl;
        }

        // Cleanup
        cap.release();
        writer.release();
        if (display) {
            cv::destroyAllWindows();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
