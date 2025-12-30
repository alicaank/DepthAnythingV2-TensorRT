#!/usr/bin/env python3
"""
Validation script for Depth Anything V2 TensorRT
Compares TensorRT inference against PyTorch reference for accuracy.
"""

import argparse
import os
import sys
import glob
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from depth_anything_v2.dpt import DepthAnythingV2


# Model configurations
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def load_pytorch_model(checkpoint_path: str, encoder: str):
    """Load PyTorch model."""
    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    return model, device


def load_tensorrt_engine(engine_path: str):
    """Load TensorRT engine."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

    # Create execution context
    context = engine.create_execution_context()

    return engine, context


def infer_tensorrt(engine, context, image: np.ndarray):
    """Run TensorRT inference."""
    import pycuda.driver as cuda

    # Allocate buffers
    h_input = cuda.pagelocked_empty(image.size, dtype=np.float32)
    h_output_shape = (1, 518, 518)
    h_output = cuda.pagelocked_empty(np.prod(h_output_shape), dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Copy input to host buffer
    np.copyto(h_input, image.ravel())

    # Transfer input to device
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    bindings = [int(d_input), int(d_output)]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer output to host
    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    # Synchronize
    stream.synchronize()

    # Reshape output
    output = h_output.reshape(h_output_shape)

    return output[0]


def validate(
    pytorch_checkpoint: str,
    tensorrt_engine: str,
    test_images_dir: str,
    encoder: str = 'vitl',
    tolerance: float = 1e-3,
    num_images: int = None
):
    """
    Validate TensorRT engine against PyTorch model.

    Args:
        pytorch_checkpoint: Path to PyTorch checkpoint
        tensorrt_engine: Path to TensorRT engine
        test_images_dir: Directory with test images
        encoder: Model encoder type
        tolerance: Maximum acceptable absolute error
        num_images: Number of images to test (None = all)
    """

    print("="*80)
    print("Depth Anything V2 TensorRT Validation")
    print("="*80)
    print(f"PyTorch checkpoint: {pytorch_checkpoint}")
    print(f"TensorRT engine: {tensorrt_engine}")
    print(f"Test images: {test_images_dir}")
    print(f"Encoder: {encoder}")
    print(f"Tolerance: {tolerance}")
    print()

    # Load models
    print("Loading PyTorch model...")
    pytorch_model, device = load_pytorch_model(pytorch_checkpoint, encoder)
    print(f"  ✓ PyTorch model loaded (device: {device})")

    print("Loading TensorRT engine...")
    trt_engine, trt_context = load_tensorrt_engine(tensorrt_engine)
    print(f"  ✓ TensorRT engine loaded")
    print()

    # Find test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(test_images_dir, '**', ext), recursive=True))

    image_files = sorted(set(image_files))

    if num_images:
        image_files = image_files[:num_images]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {test_images_dir}")

    print(f"Found {len(image_files)} test images")
    print()

    # Validation metrics
    max_errors = []
    mean_errors = []
    ssim_scores = []

    # Test each image
    for i, image_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] Testing {os.path.basename(image_path)}...", end=' ')

        # Read image
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            print("❌ Failed to read")
            continue

        # PyTorch inference
        with torch.no_grad():
            pytorch_depth = pytorch_model.infer_image(raw_image, input_size=518)

        # TensorRT inference
        # First preprocess the image (match PyTorch preprocessing)
        image, (h, w) = pytorch_model.image2tensor(raw_image, input_size=518)
        trt_input = image.cpu().numpy()

        trt_depth = infer_tensorrt(trt_engine, trt_context, trt_input)

        # Resize TensorRT output to match original size (same as PyTorch)
        trt_depth_resized = cv2.resize(trt_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Calculate metrics
        max_error = np.abs(pytorch_depth - trt_depth_resized).max()
        mean_error = np.abs(pytorch_depth - trt_depth_resized).mean()

        # Calculate SSIM
        # Normalize depths to 0-1 range for SSIM
        pytorch_norm = (pytorch_depth - pytorch_depth.min()) / (pytorch_depth.max() - pytorch_depth.min() + 1e-8)
        trt_norm = (trt_depth_resized - trt_depth_resized.min()) / (trt_depth_resized.max() - trt_depth_resized.min() + 1e-8)
        ssim_score = ssim(pytorch_norm, trt_norm, data_range=1.0)

        max_errors.append(max_error)
        mean_errors.append(mean_error)
        ssim_scores.append(ssim_score)

        status = "✓" if max_error < tolerance else "⚠"
        print(f"{status} MAE: {mean_error:.6f}, Max: {max_error:.6f}, SSIM: {ssim_score:.4f}")

    # Summary statistics
    print()
    print("="*80)
    print("Validation Summary")
    print("="*80)
    print(f"Images tested: {len(max_errors)}")
    print()
    print(f"Mean Absolute Error (MAE):")
    print(f"  Average: {np.mean(mean_errors):.6f}")
    print(f"  Std Dev: {np.std(mean_errors):.6f}")
    print(f"  Min: {np.min(mean_errors):.6f}")
    print(f"  Max: {np.max(mean_errors):.6f}")
    print()
    print(f"Maximum Absolute Error:")
    print(f"  Average: {np.mean(max_errors):.6f}")
    print(f"  Std Dev: {np.std(max_errors):.6f}")
    print(f"  Min: {np.min(max_errors):.6f}")
    print(f"  Max: {np.max(max_errors):.6f}")
    print()
    print(f"Structural Similarity (SSIM):")
    print(f"  Average: {np.mean(ssim_scores):.4f}")
    print(f"  Std Dev: {np.std(ssim_scores):.4f}")
    print(f"  Min: {np.min(ssim_scores):.4f}")
    print(f"  Max: {np.max(ssim_scores):.4f}")
    print()

    # Pass/fail
    passed = all(e < tolerance for e in max_errors)
    if passed:
        print(f"✓ PASSED: All images within tolerance ({tolerance})")
    else:
        failed_count = sum(1 for e in max_errors if e >= tolerance)
        print(f"⚠ FAILED: {failed_count}/{len(max_errors)} images exceeded tolerance ({tolerance})")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Validate TensorRT engine for Depth Anything V2')

    parser.add_argument(
        '--pytorch-checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint (.pth file)'
    )
    parser.add_argument(
        '--tensorrt-engine',
        type=str,
        required=True,
        help='Path to TensorRT engine (.trt file)'
    )
    parser.add_argument(
        '--test-images',
        type=str,
        required=True,
        help='Directory with test images'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='vitl',
        choices=['vits', 'vitb', 'vitl', 'vitg'],
        help='Model encoder type'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-3,
        help='Maximum acceptable absolute error'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Number of images to test (default: all)'
    )

    args = parser.parse_args()

    validate(
        pytorch_checkpoint=args.pytorch_checkpoint,
        tensorrt_engine=args.tensorrt_engine,
        test_images_dir=args.test_images,
        encoder=args.encoder,
        tolerance=args.tolerance,
        num_images=args.num_images
    )


if __name__ == '__main__':
    main()
