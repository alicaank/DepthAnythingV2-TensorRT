#!/bin/bash

# Complete build script for Depth Anything V2 TensorRT
# Automates ONNX export, engine building, and C++ compilation

set -e

# Configuration
ENCODER="${1:-vitl}"
PRECISION="${2:-fp16}"
BUILD_TYPE="${3:-Release}"

# Paths
CHECKPOINT_DIR="../checkpoints"
MODELS_DIR="models"
ENGINES_DIR="engines"
BUILD_DIR="build"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================="
echo "Depth Anything V2 TensorRT - Build All"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Encoder:   $ENCODER"
echo "  Precision: $PRECISION"
echo "  Build:     $BUILD_TYPE"
echo ""

# Validate encoder
if [[ ! "$ENCODER" =~ ^(vits|vitb|vitl|vitg)$ ]]; then
    echo "Error: Invalid encoder '$ENCODER'"
    echo "Usage: $0 {vits|vitb|vitl|vitg} {fp32|fp16} {Debug|Release}"
    exit 1
fi

# Validate precision
if [[ ! "$PRECISION" =~ ^(fp32|fp16)$ ]]; then
    echo "Error: Invalid precision '$PRECISION'"
    echo "Usage: $0 {vits|vitb|vitl|vitg} {fp32|fp16} {Debug|Release}"
    exit 1
fi

# Check dependencies
echo -e "${BLUE}[1/5] Checking dependencies...${NC}"

command -v python3 >/dev/null 2>&1 || { echo "Error: python3 not found"; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "Error: cmake not found"; exit 1; }
command -v make >/dev/null 2>&1 || { echo "Error: make not found"; exit 1; }

echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Create directories
mkdir -p "$MODELS_DIR" "$ENGINES_DIR"

# Checkpoint path
CHECKPOINT_PATH="${CHECKPOINT_DIR}/depth_anything_v2_${ENCODER}.pth"
ONNX_PATH="${MODELS_DIR}/${ENCODER}.onnx"
ENGINE_PATH="${ENGINES_DIR}/${ENCODER}_${PRECISION}.trt"

# Step 1: Download checkpoint if needed
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}Checkpoint not found, downloading...${NC}"
    ./scripts/download_models.sh "$ENCODER"
    echo ""
fi

# Step 2: Export to ONNX
echo -e "${BLUE}[2/5] Exporting to ONNX...${NC}"

if [ -f "$ONNX_PATH" ]; then
    echo -e "${YELLOW}ONNX model already exists, skipping export${NC}"
else
    python3 python/export_onnx.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$ONNX_PATH" \
        --encoder "$ENCODER" \
        --input-size 518 \
        --opset 17
fi

echo -e "${GREEN}✓ ONNX export complete${NC}"
echo ""

# Step 3: Build TensorRT engine
echo -e "${BLUE}[3/5] Building TensorRT engine...${NC}"

if [ -f "$ENGINE_PATH" ]; then
    echo -e "${YELLOW}TensorRT engine already exists${NC}"
    read -p "Rebuild? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping engine build"
    else
        rm "$ENGINE_PATH"
    fi
fi

if [ ! -f "$ENGINE_PATH" ]; then
    BUILD_ARGS="--onnx $ONNX_PATH --output $ENGINE_PATH --precision $PRECISION --workspace 4096"

    python3 python/build_engine.py $BUILD_ARGS
fi

echo -e "${GREEN}✓ TensorRT engine built${NC}"
echo ""

# Step 4: Build C++ code
echo -e "${BLUE}[4/5] Building C++ code...${NC}"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    ${TensorRT_ROOT:+-DTensorRT_ROOT="$TensorRT_ROOT"}

# Build
make -j$(nproc)

cd ..

echo -e "${GREEN}✓ C++ build complete${NC}"
echo ""

# Step 5: Verification
echo -e "${BLUE}[5/5] Running quick verification...${NC}"

TEST_IMAGE_URL="https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
TEST_IMAGE="data/test_image.jpg"

if [ ! -f "$TEST_IMAGE" ]; then
    mkdir -p data
    wget -q "$TEST_IMAGE_URL" -O "$TEST_IMAGE"
fi

if [ -f "${BUILD_DIR}/bin/image_inference" ]; then
    echo "Running test inference..."
    "${BUILD_DIR}/bin/image_inference" \
        --engine "$ENGINE_PATH" \
        --input "$TEST_IMAGE" \
        --output "data/test_output.png"

    echo -e "${GREEN}✓ Verification complete${NC}"
else
    echo -e "${YELLOW}Warning: image_inference binary not found, skipping verification${NC}"
fi

echo ""
echo "========================================="
echo -e "${GREEN}BUILD COMPLETE!${NC}"
echo "========================================="
echo ""
echo "Generated files:"
echo "  ONNX:   $ONNX_PATH"
echo "  Engine: $ENGINE_PATH"
echo "  Binary: ${BUILD_DIR}/bin/"
echo ""
echo "Next steps:"
echo "  # Run image inference"
echo "  ${BUILD_DIR}/bin/image_inference --engine $ENGINE_PATH --input <image>"
echo ""
echo "  # Run video inference"
echo "  ${BUILD_DIR}/bin/video_inference --engine $ENGINE_PATH --input <video> --display"
echo ""
echo "  # Run benchmark"
echo "  ${BUILD_DIR}/bin/benchmark --engine $ENGINE_PATH"
echo ""
