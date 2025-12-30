# Depth Anything V2 - TensorRT C++ Implementation

High-performance TensorRT implementation of Depth Anything V2 for real-time depth estimation.

## Repository Notes

This folder contains both C++/CUDA inference code and Python helper tools (ONNX export, engine build, validation).

Generated artifacts (TensorRT engines, ONNX models, build outputs, and benchmarking/visualization outputs) are intentionally ignored by git via `.gitignore`.

## Performance

**Target Performance on RTX 5070/5060 (ViT-L model):**

| GPU | Precision | Latency | FPS |
|-----|-----------|---------|-----|
| RTX 5070 | FP16 | ~22ms | 30-35 |

## Features

- ✅ **FP16 Precision** - Optimized precision mode
- ✅ **CUDA Graph Optimization** - Reduced kernel launch overhead
- ✅ **GPU Preprocessing** - Fused resize+normalize kernel
- ✅ **Multi-Stream Pipeline** - Overlapped computation for video
- ✅ **PyTorch Consistency** - Exact preprocessing/postprocessing match
- ✅ **Production Ready** - RAII memory management, error handling

## Prerequisites

### System Requirements

- CUDA 11.8+ (12.x recommended for RTX 50xx series)
- TensorRT 8.6+ (10.x recommended)
- cuDNN 8.9+
- OpenCV 4.x
- CMake 3.18+
- GCC 9+ or Clang 10+

### Installation

#### Ubuntu/Debian

```bash
# Install CUDA and TensorRT (follow NVIDIA official guides)
# https://developer.nvidia.com/cuda-downloads
# https://developer.nvidia.com/tensorrt

# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    python3-pip

# Install Python dependencies
pip3 install -r requirements.txt
```

## Quick Start

### Step 1: Export ONNX Model

```bash
# Download model checkpoint (if not already present)
cd ..
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth \
    -P checkpoints/

# Export to ONNX
cd tensorrt
python python/export_onnx.py \
    --encoder vitl \
    --checkpoint ../checkpoints/depth_anything_v2_vitl.pth \
    --output models/vitl.onnx
```

### Step 2: Build TensorRT Engine

```bash
# FP16 engine (recommended starting point)
python python/build_engine.py \
    --onnx models/vitl.onnx \
    --output engines/vitl_fp16.trt \
    --precision fp16 \
    --workspace 4096
```

### Step 3: Build C++ Code

```bash
mkdir build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTensorRT_ROOT=/usr/local/tensorrt  # Adjust path if needed

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### Step 4: Run Examples

```bash
# Image inference
./bin/image_inference \
    --engine ../engines/vitl_fp16.trt \
    --input /path/to/image.jpg \
    --output depth_output.png \
    --colormap

# Video inference
./bin/video_inference \
    --engine ../engines/vitl_fp16.trt \
    --input /path/to/video.mp4 \
    --output depth_video.mp4 \
    --colormap \
    --display

# Benchmark
./bin/benchmark \
    --engine ../engines/vitl_fp16.trt \
    --iterations 1000
```

### Step 5: Compare PyTorch vs TensorRT (sanity check)

This runs PyTorch as a reference and invokes the C++ TensorRT binary to compare outputs.

```bash
python3 python/compare_outputs.py \
    --image ../assets/examples/demo03.jpg \
    --checkpoint ../checkpoints/depth_anything_v2_vitl.pth \
    --trt-engine engines/vitl_fp16_new.trt \
    --encoder vitl \
    --save-dir comparison_results
```

### Step 6: Benchmark

```bash
./build/bin/benchmark --engine engines/vitl_fp16_new.trt
```

## Output Examples

### Image Inference (`image_inference`)

**Command:**

```bash
./bin/image_inference --engine engines/vitl_fp16.trt --input image.jpg --output depth_output.png --colormap
```

**Files created:**
- `depth_output.png`

**Example console output (abridged):**

```text
========================================
Depth Anything V2 - Image Inference
========================================
Engine: engines/vitl_fp16.trt
Input: image.jpg
Output: depth_output.png
Device: 0
Colormap: Yes

Loading input image...
  Image size: 1920x1080

Running inference...
  ✓ Inference complete

Saving output...
  ✓ Saved to: depth_output.png

Performance Statistics:
  Preprocessing: ... ms
  Inference:     ... ms
  Postprocessing: ... ms
  Total:         ... ms
  FPS:           ...

✓ Complete!
```

### PyTorch vs TensorRT Comparison (`compare_outputs.py`)

**Command:**

```bash
python3 python/compare_outputs.py \
    --image ../assets/examples/demo03.jpg \
    --checkpoint ../checkpoints/depth_anything_v2_vitl.pth \
    --trt-engine engines/vitl_fp16_new.trt \
    --encoder vitl \
    --save-dir comparison_results
```

**Files created (under `comparison_results/`):**
- `tensorrt_raw_output.png`
- `pytorch_depth.png`
- `tensorrt_depth.png`
- `difference_map.png`

**Example console output (abridged):**

```text
============================================================
Depth Anything V2 - PyTorch vs TensorRT Comparison
============================================================
Image:      ../assets/examples/demo03.jpg
Checkpoint: ../checkpoints/depth_anything_v2_vitl.pth
Engine:     engines/vitl_fp16_new.trt
Encoder:    vitl
Device:     cuda

Running PyTorch inference...
  ✓ PyTorch inference complete

Running TensorRT inference...
  ✓ TensorRT inference complete

============================================================
COMPARISON RESULTS
============================================================
Statistical Metrics:
  Correlation:        ...
  RMSE:               ...
  MAE:                ...
  Mean Rel Error:     ...%

✓ Saved visualizations to: comparison_results

✓ Comparison complete!
```

### Benchmark (`benchmark`)

**Command:**

```bash
./bin/benchmark --engine engines/vitl_fp16.trt --iterations 1000
```

**Example console output (abridged):**

```text
========================================
Depth Anything V2 - Benchmark Tool
========================================
Engine: engines/vitl_fp16.trt
Device: 0
Input size: 1920x1080
Warmup iterations: 100
Benchmark iterations: 1000

Warming up (100 iterations)...

Running benchmark (1000 iterations)...

========================================
BENCHMARK RESULTS
========================================

End-to-End Results:
  Latency (ms):
    Min:    ...
    Mean:   ...
    Median: ...
    P95:    ...
    P99:    ...
    Max:    ...
    Std:    ...
  Throughput: ... FPS
```

## Detailed Usage

### Python Tools

#### 1. ONNX Export

```bash
python python/export_onnx.py \
    --checkpoint <path_to_pth> \
    --output <path_to_onnx> \
    --encoder {vits|vitb|vitl|vitg} \
    --input-size 518 \
    --opset 17
```

**Options:**
- `--checkpoint`: PyTorch checkpoint file (.pth)
- `--output`: Output ONNX file path
- `--encoder`: Model size (vits, vitb, vitl, vitg)
- `--input-size`: Input resolution (must be multiple of 14)
- `--opset`: ONNX opset version (17 recommended)
- `--no-simplify`: Disable ONNX simplification
- `--verbose`: Enable verbose logging

#### 2. TensorRT Engine Building

```bash
python python/build_engine.py \
    --onnx <path_to_onnx> \
    --output <path_to_engine> \
    --precision {fp32|fp16} \
    --workspace 4096
```

**Options:**
- `--onnx`: Input ONNX model
- `--output`: Output TensorRT engine file
- `--precision`: Precision mode (fp32/fp16)
- `--workspace`: Max workspace size in MB (4096 recommended)
- `--max-batch-size`: Maximum batch size (default: 1)
- `--verbose`: Enable verbose logging

#### 3. Validation

```bash
python python/validate.py \
    --pytorch-checkpoint ../checkpoints/depth_anything_v2_vitl.pth \
    --tensorrt-engine engines/vitl_fp16.trt \
    --test-images /path/to/test/images \
    --encoder vitl \
    --tolerance 1e-3
```

### C++ Examples

#### Image Inference

```bash
./bin/image_inference \
    --engine engines/vitl_fp16.trt \
    --input image.jpg \
    --output depth.png \
    --colormap \
    --device 0
```

#### Video Inference

```bash
# Process video file
./bin/video_inference \
    --engine engines/vitl_fp16.trt \
    --input video.mp4 \
    --output depth_video.mp4 \
    --colormap \
    --display
```

#### Benchmark

```bash
./bin/benchmark \
    --engine engines/vitl_fp16.trt \
    --iterations 1000 \
    --warmup 100 \
    --width 1920 \
    --height 1080 \
    --device 0
```

## API Usage

### C++ API

```cpp
#include "depth_anything_v2.hpp"

using namespace depth_anything_v2;

// Initialize engine
InferenceConfig config;
config.device_id = 0;
config.enable_cuda_graph = true;
config.use_pinned_memory = true;

DepthAnythingV2 engine("engines/vitl_fp16.trt", config);

// Load image
cv::Mat input = cv::imread("image.jpg");

// Run inference
cv::Mat depth = engine.infer(input);

// With visualization
cv::Mat depth, depth_vis;
engine.inferWithVisualization(input, depth, depth_vis);

// Get statistics
auto stats = engine.getStatistics();
std::cout << "FPS: " << (1000.0f / stats.avg_total_ms) << std::endl;
```

## Performance Optimization Guide

### 1. Choose the Right Precision

- **FP32**: Baseline, highest accuracy
- **FP16**: 2-3x faster, minimal accuracy loss (<0.1%)

**Recommendation:** Start with FP16.

### 2. Enable CUDA Graph

CUDA Graph reduces kernel launch overhead by ~10-15%:

```cpp
InferenceConfig config;
config.enable_cuda_graph = true;  // Enabled by default
```

### 3. Use Pinned Memory

Faster CPU-GPU transfers:

```cpp
config.use_pinned_memory = true;  // Enabled by default
```

### 4. Profile Your Application

```bash
# Profile with Nsight Systems
nsys profile -o profile.nsys-rep ./bin/benchmark --engine engines/vitl_fp16.trt

# View with Nsight Systems GUI
nsight-sys profile.nsys-rep
```

### 5. Reduce Input Resolution

If 518×518 is too slow, try smaller sizes (must be multiple of 14):

- 476×476: ~15% faster
- 448×448: ~25% faster
- 420×420: ~35% faster
- 392×392: ~45% faster

## Troubleshooting

### Issue: Low FPS

**Check:**
1. GPU utilization: `nvidia-smi dmon -s u` (should be >90%)
2. TensorRT engine is used (not ONNX Runtime)
3. CUDA Graph is enabled
4. Input images aren't too large

**Solutions:**
- Reduce input resolution
- Enable all optimizations

### Issue: Accuracy Degradation

**For FP16:**
- Should have <0.1% MAE difference
- If higher, check preprocessing

### Issue: Build Errors

**TensorRT not found:**
```bash
export TensorRT_ROOT=/path/to/tensorrt
cmake .. -DTensorRT_ROOT=$TensorRT_ROOT
```

**CUDA not found:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```



## License

This implementation is distributed under the repository's Apache-2.0 license.

Model weights are subject to their own licensing terms; see the main repository README for details.

## Citation

If you use this implementation, please cite:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

## Support

For issues specific to this TensorRT implementation:
- Check the troubleshooting section above
- Review NVIDIA TensorRT documentation
- Open an issue with profiling data

For general Depth Anything V2 questions:
- See main repository: https://github.com/DepthAnything/Depth-Anything-V2
