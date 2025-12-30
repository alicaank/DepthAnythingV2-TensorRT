#!/usr/bin/env python3
"""
Compare TensorRT and PyTorch outputs for Depth Anything V2
Validates that TensorRT inference matches PyTorch reference implementation
"""

import argparse
import sys
import os
import numpy as np
import cv2
import torch

# Add parent directory to path for depth_anything_v2 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from depth_anything_v2.dpt import DepthAnythingV2


# Model configurations
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def load_pytorch_model(checkpoint_path, encoder='vitl', device='cuda', use_half=True):
    """Load PyTorch model"""
    print(f"\nLoading PyTorch model ({encoder})...")

    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Use half precision if requested (required for xformers on newer GPUs)
    if use_half and device == 'cuda':
        model = model.half()
        print(f"  ✓ Using FP16 precision")

    print(f"  ✓ Model loaded successfully")
    return model


def preprocess_image(image, target_size=518):
    """Preprocess image to match TensorRT C++ preprocessing

    This matches the fused CUDA kernel which does:
    1. Direct bilinear resize to target_size x target_size (no padding!)
    2. BGR to RGB conversion
    3. Normalization with ImageNet mean/std
    4. HWC to CHW format conversion
    """
    # Direct resize to target size (matches C++ preprocessing)
    # INTER_LINEAR matches the bilinear interpolation in CUDA kernel
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return resized


def pytorch_inference(model, image_path, device='cuda', use_half=True):
    """Run PyTorch inference"""
    print(f"\nRunning PyTorch inference...")

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    orig_h, orig_w = image.shape[:2]
    print(f"  Input size: {orig_w}x{orig_h}")

    # Preprocess
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = preprocess_image(image_rgb, 518)

    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Move to device and convert to correct dtype BEFORE normalization
    dtype = torch.float16 if use_half and device == 'cuda' else torch.float32
    image_tensor = image_tensor.to(device, dtype=dtype)
    mean = mean.to(device, dtype=dtype)
    std = std.to(device, dtype=dtype)

    # Normalize
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        depth = model(image_tensor)

    # Extract depth map (output is [1, H, W])
    depth_map = depth.squeeze().cpu().float().numpy()

    # Resize to original dimensions
    depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    print(f"  ✓ PyTorch inference complete")
    print(f"  Output shape: {depth_map.shape}")
    print(f"  Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

    return depth_map


def load_tensorrt_output(output_path):
    """Load TensorRT output depth map"""
    print(f"\nLoading TensorRT output from: {output_path}")

    # TensorRT saves as grayscale visualization, need to load raw depth
    # For now, assume we saved the raw depth map
    if output_path.endswith('.npy'):
        depth_map = np.load(output_path)
    else:
        # If PNG, load as grayscale and denormalize
        depth_map = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        if depth_map is None:
            raise ValueError(f"Failed to load TensorRT output: {output_path}")
        depth_map = depth_map.astype(np.float32)

    print(f"  ✓ TensorRT output loaded")
    print(f"  Output shape: {depth_map.shape}")
    print(f"  Depth range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

    return depth_map


def compare_outputs(pytorch_depth, tensorrt_depth, save_dir=None):
    """Compare PyTorch and TensorRT outputs"""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    # Ensure same shape
    if pytorch_depth.shape != tensorrt_depth.shape:
        print(f"⚠ Warning: Shape mismatch!")
        print(f"  PyTorch: {pytorch_depth.shape}")
        print(f"  TensorRT: {tensorrt_depth.shape}")

        # Resize TensorRT to match PyTorch
        tensorrt_depth = cv2.resize(
            tensorrt_depth,
            (pytorch_depth.shape[1], pytorch_depth.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        print(f"  Resized TensorRT to match PyTorch")

    # Flatten for easier comparison
    pt_flat = pytorch_depth.flatten()
    trt_flat = tensorrt_depth.flatten()

    # Compute metrics
    abs_diff = np.abs(pt_flat - trt_flat)
    rel_diff = abs_diff / (np.abs(pt_flat) + 1e-6)

    # Correlation
    correlation = np.corrcoef(pt_flat, trt_flat)[0, 1]

    # Mean Squared Error
    mse = np.mean((pt_flat - trt_flat) ** 2)
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(abs_diff)

    # Relative Error
    mean_rel_error = np.mean(rel_diff) * 100  # in percentage

    print(f"\nStatistical Metrics:")
    print(f"  Correlation:        {correlation:.6f}")
    print(f"  RMSE:               {rmse:.6f}")
    print(f"  MAE:                {mae:.6f}")
    print(f"  Mean Rel Error:     {mean_rel_error:.4f}%")

    print(f"\nAbsolute Difference:")
    print(f"  Min:     {abs_diff.min():.6f}")
    print(f"  Mean:    {abs_diff.mean():.6f}")
    print(f"  Median:  {np.median(abs_diff):.6f}")
    print(f"  Max:     {abs_diff.max():.6f}")
    print(f"  Std:     {abs_diff.std():.6f}")

    print(f"\nRelative Difference (%):")
    print(f"  Mean:    {np.mean(rel_diff) * 100:.4f}%")
    print(f"  Median:  {np.median(rel_diff) * 100:.4f}%")
    print(f"  P95:     {np.percentile(rel_diff, 95) * 100:.4f}%")
    print(f"  P99:     {np.percentile(rel_diff, 99) * 100:.4f}%")

    # Accuracy assessment
    print(f"\n{'='*60}")
    if correlation > 0.999:
        print("✅ EXCELLENT: TensorRT output matches PyTorch very closely!")
    elif correlation > 0.995:
        print("✅ GOOD: TensorRT output matches PyTorch well")
    elif correlation > 0.99:
        print("⚠  ACCEPTABLE: Some differences detected")
    else:
        print("❌ WARNING: Significant differences detected!")
    print(f"{'='*60}")

    # Save visualizations if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Normalize for visualization
        def normalize_depth(d):
            d_min, d_max = d.min(), d.max()
            return ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # Save individual depth maps
        cv2.imwrite(
            os.path.join(save_dir, 'pytorch_depth.png'),
            normalize_depth(pytorch_depth)
        )
        cv2.imwrite(
            os.path.join(save_dir, 'tensorrt_depth.png'),
            normalize_depth(tensorrt_depth)
        )

        # Save difference map
        diff_map = np.abs(pytorch_depth - tensorrt_depth)
        diff_normalized = normalize_depth(diff_map)
        diff_colored = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(
            os.path.join(save_dir, 'difference_map.png'),
            diff_colored
        )

        print(f"\n✓ Saved visualizations to: {save_dir}")

    return {
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae,
        'mean_rel_error': mean_rel_error
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare TensorRT and PyTorch outputs for Depth Anything V2'
    )

    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Input image path'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='PyTorch checkpoint path (.pth)'
    )
    parser.add_argument(
        '--trt-engine',
        type=str,
        required=True,
        help='TensorRT engine path (.trt)'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='vitl',
        choices=['vits', 'vitb', 'vitl', 'vitg'],
        help='Model encoder type'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='comparison_results',
        help='Directory to save comparison visualizations'
    )
    parser.add_argument(
        '--cpp-inference',
        type=str,
        default='./build/bin/image_inference',
        help='Path to C++ inference binary'
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not os.path.exists(args.trt_engine):
        print(f"Error: TensorRT engine not found: {args.trt_engine}")
        sys.exit(1)

    print("="*60)
    print("Depth Anything V2 - PyTorch vs TensorRT Comparison")
    print("="*60)
    print(f"Image:      {args.image}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Engine:     {args.trt_engine}")
    print(f"Encoder:    {args.encoder}")
    print(f"Device:     {args.device}")

    # 1. Run PyTorch inference
    model = load_pytorch_model(args.checkpoint, args.encoder, args.device)
    pytorch_depth = pytorch_inference(model, args.image, args.device)

    # 2. Run TensorRT inference
    print(f"\nRunning TensorRT inference...")
    trt_output = os.path.join(args.save_dir, 'tensorrt_raw_output.png')
    os.makedirs(args.save_dir, exist_ok=True)

    # Run C++ inference
    import subprocess
    result = subprocess.run([
        args.cpp_inference,
        '--engine', args.trt_engine,
        '--input', args.image,
        '--output', trt_output,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running TensorRT inference:")
        print(result.stderr)
        sys.exit(1)

    print(f"  ✓ TensorRT inference complete")

    # Load TensorRT output
    tensorrt_depth = cv2.imread(trt_output, cv2.IMREAD_GRAYSCALE)
    if tensorrt_depth is None:
        print(f"Error: Failed to load TensorRT output")
        sys.exit(1)

    tensorrt_depth = tensorrt_depth.astype(np.float32)

    # Normalize both to same range for fair comparison
    def normalize_to_range(depth):
        d_min, d_max = depth.min(), depth.max()
        return (depth - d_min) / (d_max - d_min)

    pytorch_depth_norm = normalize_to_range(pytorch_depth)
    tensorrt_depth_norm = normalize_to_range(tensorrt_depth)

    # 3. Compare outputs
    metrics = compare_outputs(
        pytorch_depth_norm,
        tensorrt_depth_norm,
        args.save_dir
    )

    print(f"\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
