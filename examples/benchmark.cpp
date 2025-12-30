/**
 * Benchmark Tool for Depth Anything V2
 *
 * Comprehensive performance profiling with detailed metrics
 *
 * Usage:
 *   ./benchmark --engine <path_to_engine> [--iterations <num>]
 */

#include "depth_anything_v2.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>

using namespace depth_anything_v2;

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --engine <path>        Path to TensorRT engine file (required)\n"
              << "  --iterations <num>     Number of benchmark iterations (default: 1000)\n"
              << "  --warmup <num>         Number of warmup iterations (default: 100)\n"
              << "  --width <pixels>       Input image width (default: 1920)\n"
              << "  --height <pixels>      Input image height (default: 1080)\n"
              << "  --device <id>          CUDA device ID (default: 0)\n"
              << "  --help                 Show this help message\n"
              << std::endl;
}

struct BenchmarkResults {
    std::vector<float> latencies;
    float min_latency;
    float max_latency;
    float mean_latency;
    float median_latency;
    float p95_latency;
    float p99_latency;
    float std_dev;
    float throughput_fps;
};

BenchmarkResults analyzeBenchmark(const std::vector<float>& latencies) {
    BenchmarkResults results;
    results.latencies = latencies;

    if (latencies.empty()) {
        throw std::runtime_error("No latency data");
    }

    // Sort for percentile calculations
    std::vector<float> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());

    // Min/Max
    results.min_latency = sorted_latencies.front();
    results.max_latency = sorted_latencies.back();

    // Mean
    float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
    results.mean_latency = sum / latencies.size();

    // Median
    size_t mid = sorted_latencies.size() / 2;
    if (sorted_latencies.size() % 2 == 0) {
        results.median_latency = (sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2.0f;
    } else {
        results.median_latency = sorted_latencies[mid];
    }

    // P95
    size_t p95_idx = static_cast<size_t>(sorted_latencies.size() * 0.95);
    results.p95_latency = sorted_latencies[p95_idx];

    // P99
    size_t p99_idx = static_cast<size_t>(sorted_latencies.size() * 0.99);
    results.p99_latency = sorted_latencies[p99_idx];

    // Standard deviation
    float variance = 0.0f;
    for (float latency : latencies) {
        float diff = latency - results.mean_latency;
        variance += diff * diff;
    }
    variance /= latencies.size();
    results.std_dev = std::sqrt(variance);

    // Throughput (FPS)
    results.throughput_fps = 1000.0f / results.mean_latency;

    return results;
}

void printResults(const std::string& name, const BenchmarkResults& results) {
    std::cout << "\n" << name << " Results:" << std::endl;
    std::cout << "  Latency (ms):" << std::endl;
    std::cout << "    Min:    " << std::fixed << std::setprecision(3) << results.min_latency << std::endl;
    std::cout << "    Mean:   " << results.mean_latency << std::endl;
    std::cout << "    Median: " << results.median_latency << std::endl;
    std::cout << "    P95:    " << results.p95_latency << std::endl;
    std::cout << "    P99:    " << results.p99_latency << std::endl;
    std::cout << "    Max:    " << results.max_latency << std::endl;
    std::cout << "    Std:    " << results.std_dev << std::endl;
    std::cout << "  Throughput: " << std::setprecision(2) << results.throughput_fps << " FPS" << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string engine_path;
    int iterations = 1000;
    int warmup_iterations = 100;
    int image_width = 1920;
    int image_height = 1080;
    int device_id = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            image_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            image_height = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate arguments
    if (engine_path.empty()) {
        std::cerr << "Error: --engine is required\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Depth Anything V2 - Benchmark Tool" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Engine: " << engine_path << std::endl;
        std::cout << "Device: " << device_id << std::endl;
        std::cout << "Input size: " << image_width << "x" << image_height << std::endl;
        std::cout << "Warmup iterations: " << warmup_iterations << std::endl;
        std::cout << "Benchmark iterations: " << iterations << std::endl;
        std::cout << std::endl;

        // Initialize inference engine
        std::cout << "Initializing engine..." << std::endl;
        InferenceConfig config;
        config.device_id = device_id;
        config.enable_cuda_graph = true;
        config.use_pinned_memory = true;

        DepthAnythingV2 engine(engine_path, config);
        std::cout << std::endl;

        // Create test image
        cv::Mat test_image(image_height, image_width, CV_8UC3);
        cv::randu(test_image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        // Warmup
        std::cout << "Warming up (" << warmup_iterations << " iterations)..." << std::endl;
        for (int i = 0; i < warmup_iterations; ++i) {
            engine.infer(test_image);

            if ((i + 1) % 10 == 0) {
                std::cout << "\r  Progress: " << (i + 1) << "/" << warmup_iterations << std::flush;
            }
        }
        std::cout << std::endl;
        engine.resetStatistics();

        // Benchmark
        std::cout << "\nRunning benchmark (" << iterations << " iterations)..." << std::endl;
        std::vector<float> total_latencies;
        std::vector<float> preprocess_latencies;
        std::vector<float> inference_latencies;
        std::vector<float> postprocess_latencies;

        auto benchmark_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            engine.infer(test_image);
            auto end = std::chrono::high_resolution_clock::now();

            float latency = std::chrono::duration<float, std::milli>(end - start).count();
            total_latencies.push_back(latency);

            auto stats = engine.getStatistics();
            preprocess_latencies.push_back(stats.avg_preprocess_ms);
            inference_latencies.push_back(stats.avg_inference_ms);
            postprocess_latencies.push_back(stats.avg_postprocess_ms);

            if ((i + 1) % 100 == 0) {
                std::cout << "\r  Progress: " << (i + 1) << "/" << iterations << std::flush;
            }
        }

        auto benchmark_end = std::chrono::high_resolution_clock::now();
        float total_time = std::chrono::duration<float>(benchmark_end - benchmark_start).count();

        std::cout << std::endl;
        std::cout << "\nBenchmark complete!" << std::endl;
        std::cout << "Total time: " << total_time << " seconds" << std::endl;

        // Analyze results
        auto total_results = analyzeBenchmark(total_latencies);
        auto preprocess_results = analyzeBenchmark(preprocess_latencies);
        auto inference_results = analyzeBenchmark(inference_latencies);
        auto postprocess_results = analyzeBenchmark(postprocess_latencies);

        // Print results
        std::cout << "\n========================================" << std::endl;
        std::cout << "BENCHMARK RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;

        printResults("End-to-End", total_results);
        printResults("Preprocessing", preprocess_results);
        printResults("Inference", inference_results);
        printResults("Postprocessing", postprocess_results);

        // Summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "SUMMARY" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Average Latency: " << std::fixed << std::setprecision(2)
                  << total_results.mean_latency << " ms" << std::endl;
        std::cout << "Throughput:      " << total_results.throughput_fps << " FPS" << std::endl;
        std::cout << std::endl;

        std::cout << "Breakdown (% of total):" << std::endl;
        float total = preprocess_results.mean_latency +
                     inference_results.mean_latency +
                     postprocess_results.mean_latency;
        std::cout << "  Preprocessing:  " << std::setprecision(1)
                  << (preprocess_results.mean_latency / total * 100) << "%" << std::endl;
        std::cout << "  Inference:      "
                  << (inference_results.mean_latency / total * 100) << "%" << std::endl;
        std::cout << "  Postprocessing: "
                  << (postprocess_results.mean_latency / total * 100) << "%" << std::endl;
        std::cout << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
