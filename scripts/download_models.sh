#!/bin/bash

# Download Depth Anything V2 Model Checkpoints
# Downloads pretrained weights from HuggingFace

set -e

CHECKPOINT_DIR="../checkpoints"
BASE_URL="https://huggingface.co/depth-anything"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Depth Anything V2 - Model Downloader"
echo "========================================="
echo ""

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Function to download model
download_model() {
    local model_name=$1
    local model_file=$2
    local model_url="${BASE_URL}/${model_name}/resolve/main/${model_file}"
    local output_path="${CHECKPOINT_DIR}/${model_file}"

    if [ -f "$output_path" ]; then
        echo -e "${YELLOW}✓ $model_file already exists, skipping${NC}"
    else
        echo -e "${GREEN}Downloading $model_file...${NC}"
        wget -q --show-progress "$model_url" -O "$output_path"
        echo -e "${GREEN}✓ Downloaded $model_file${NC}"
    fi
}

# Parse arguments
MODEL_SIZE="${1:-all}"

case "$MODEL_SIZE" in
    vits)
        echo "Downloading ViT-Small model..."
        download_model "Depth-Anything-V2-Small" "depth_anything_v2_vits.pth"
        ;;
    vitb)
        echo "Downloading ViT-Base model..."
        download_model "Depth-Anything-V2-Base" "depth_anything_v2_vitb.pth"
        ;;
    vitl)
        echo "Downloading ViT-Large model..."
        download_model "Depth-Anything-V2-Large" "depth_anything_v2_vitl.pth"
        ;;
    vitg)
        echo "Downloading ViT-Giant model..."
        download_model "Depth-Anything-V2-Giant" "depth_anything_v2_vitg.pth"
        ;;
    all)
        echo "Downloading all models..."
        download_model "Depth-Anything-V2-Small" "depth_anything_v2_vits.pth"
        download_model "Depth-Anything-V2-Base" "depth_anything_v2_vitb.pth"
        download_model "Depth-Anything-V2-Large" "depth_anything_v2_vitl.pth"
        download_model "Depth-Anything-V2-Giant" "depth_anything_v2_vitg.pth"
        ;;
    *)
        echo "Usage: $0 {vits|vitb|vitl|vitg|all}"
        echo ""
        echo "  vits  - ViT-Small (fastest, lowest quality)"
        echo "  vitb  - ViT-Base"
        echo "  vitl  - ViT-Large (recommended)"
        echo "  vitg  - ViT-Giant (highest quality, slowest)"
        echo "  all   - Download all models"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ Download complete!${NC}"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
