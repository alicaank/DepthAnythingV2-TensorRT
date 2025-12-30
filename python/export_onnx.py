#!/usr/bin/env python3
"""
ONNX Export Script for Depth Anything V2
Converts PyTorch checkpoint to ONNX format for TensorRT optimization.

This script ensures proper intermediate layer extraction and model architecture
preservation for TensorRT compatibility.
"""

import argparse
import os
import sys
import torch
import torch.onnx
import onnx
from onnx import shape_inference

# Add parent directory to path to import depth_anything_v2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from depth_anything_v2.dpt import DepthAnythingV2


# Model configurations from run.py lines 28-33
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    encoder: str = 'vitl',
    input_size: int = 518,
    opset_version: int = 17,
    simplify: bool = True,
    verbose: bool = False
):
    """
    Export Depth Anything V2 model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth file)
        output_path: Path to save ONNX model
        encoder: Model encoder type (vits, vitb, vitl, vitg)
        input_size: Input image size (must be multiple of 14)
        opset_version: ONNX opset version (17 recommended for TensorRT)
        simplify: Whether to simplify the ONNX model
        verbose: Enable verbose logging
    """

    # Validate encoder
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Invalid encoder: {encoder}. Must be one of {list(MODEL_CONFIGS.keys())}")

    # Validate input size (must be multiple of 14 for patch embedding)
    if input_size % 14 != 0:
        raise ValueError(f"Input size must be multiple of 14, got {input_size}")

    print(f"Exporting Depth Anything V2 ({encoder}) to ONNX...")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Opset version: {opset_version}")

    # Create model
    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # Replace xformers attention with standard PyTorch attention for export
    # This is necessary because xformers doesn't support ONNX export
    print("Replacing xformers attention with standard PyTorch attention for ONNX export...")

    def replace_attention(module):
        """Replace MemEffAttention with standard PyTorch attention"""
        for name, child in module.named_children():
            if child.__class__.__name__ == 'MemEffAttention':
                # Replace with standard attention
                import torch.nn.functional as F

                class StandardAttention(torch.nn.Module):
                    def __init__(self, orig_module):
                        super().__init__()
                        self.num_heads = orig_module.num_heads
                        self.qkv = orig_module.qkv
                        self.attn_drop = orig_module.attn_drop
                        self.proj = orig_module.proj
                        self.proj_drop = orig_module.proj_drop

                    def forward(self, x):
                        B, N, C = x.shape
                        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv[0], qkv[1], qkv[2]

                        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.num_heads) ** 0.5)
                        attn = F.softmax(attn, dim=-1)
                        attn = self.attn_drop(attn)

                        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                        x = self.proj(x)
                        x = self.proj_drop(x)
                        return x

                setattr(module, name, StandardAttention(child))
            else:
                replace_attention(child)

    replace_attention(model)

    # Move to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create dummy input on the same device
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # Export to ONNX (use legacy path to avoid torch.export issues)
    print(f"Exporting to ONNX on {device}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['depth'],
            dynamic_axes={
                'image': {0: 'batch'},
                'depth': {0: 'batch'}
            },
            verbose=verbose,
            # Use legacy exporter to avoid torch.export issues with xformers
            dynamo=False
        )

    print(f"ONNX model exported to {output_path}")

    # Load and verify the ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Apply shape inference
    print("Applying shape inference...")
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(onnx_model, output_path)

    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print("Simplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, output_path)
                print("ONNX model simplified successfully!")
            else:
                print("Warning: ONNX simplification check failed, using unsimplified model")
        except ImportError:
            print("Warning: onnx-simplifier not installed, skipping simplification")
            print("Install with: pip install onnx-simplifier")

    # Print model info
    print("\nModel Information:")
    print(f"  IR Version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Opset Version: {onnx_model.opset_import[0].version}")
    print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
    print(f"  Number of nodes: {len(onnx_model.graph.node)}")

    # Validate output with PyTorch
    print("\nValidating ONNX output against PyTorch...")
    import onnxruntime as ort
    import numpy as np

    # PyTorch inference
    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()

    # ONNX inference (use CUDA provider if available)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(output_path, providers=providers)
    onnx_out = ort_session.run(None, {'image': dummy_input.cpu().numpy()})[0]

    # Compare outputs
    max_diff = np.abs(torch_out - onnx_out).max()
    mean_diff = np.abs(torch_out - onnx_out).mean()

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-4:
        print("  ✓ ONNX model matches PyTorch model!")
    else:
        print(f"  ⚠ Warning: Large difference detected ({max_diff:.6f})")

    print(f"\nExport complete! ONNX model saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Export Depth Anything V2 to ONNX')

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint (.pth file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save ONNX model (.onnx file)'
    )
    parser.add_argument(
        '--encoder',
        type=str,
        default='vitl',
        choices=['vits', 'vitb', 'vitl', 'vitg'],
        help='Model encoder type'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=518,
        help='Input image size (must be multiple of 14)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX model simplification'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Export to ONNX
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        encoder=args.encoder,
        input_size=args.input_size,
        opset_version=args.opset,
        simplify=not args.no_simplify,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
