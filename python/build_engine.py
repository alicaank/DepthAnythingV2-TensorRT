#!/usr/bin/env python3
"""
TensorRT Engine Builder for Depth Anything V2
Builds optimized TensorRT engine from ONNX model with FP16 support.
"""

import argparse
import os
import sys
import tensorrt as trt

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = 'fp16',
    workspace: int = 4096,
    max_batch_size: int = 1,
    verbose: bool = False
):
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16')
        workspace: Max workspace size in MB
        max_batch_size: Maximum batch size
        verbose: Enable verbose logging
    """

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if precision not in ['fp32', 'fp16']:
        raise ValueError(f"Invalid precision: {precision}. Must be 'fp32' or 'fp16'")

    print(f"Building TensorRT engine...")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Output engine: {engine_path}")
    print(f"  Precision: {precision.upper()}")
    print(f"  Workspace: {workspace} MB")
    print(f"  Max batch size: {max_batch_size}")

    # Set logger level
    if verbose:
        TRT_LOGGER.min_severity = trt.Logger.VERBOSE
    else:
        TRT_LOGGER.min_severity = trt.Logger.INFO

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

    print(f"  Network inputs: {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"  Network outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 20))

    # Set precision mode
    if precision == 'fp16':
        print("Enabling FP16 precision...")
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  ✓ FP16 is supported on this platform")
        else:
            print("  ⚠ Warning: FP16 is not well-supported on this platform")
            config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile
    print("Setting optimization profile...")
    profile = builder.create_optimization_profile()

    # Get input shape from network
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # Set optimization profile (fixed batch size for optimal performance)
    # Shape: [batch, channels, height, width]
    min_shape = (1, 3, 518, 518)
    opt_shape = (max_batch_size, 3, 518, 518)
    max_shape = (max_batch_size, 3, 518, 518)

    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"  Optimization profile: {opt_shape}")

    # Build engine
    print("Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build TensorRT engine")
        sys.exit(1)

    # Save engine
    print(f"Saving engine to {engine_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"\n✓ TensorRT engine built successfully!")
    print(f"  Engine file: {engine_path}")
    print(f"  File size: {os.path.getsize(engine_path) / 1024 / 1024:.2f} MB")

    # Verify engine
    print("\nVerifying engine...")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    if engine is None:
        print("ERROR: Failed to deserialize engine")
        sys.exit(1)

    print(f"  ✓ Engine verification passed!")

    # Use new TensorRT 10+ API
    num_io_tensors = engine.num_io_tensors
    print(f"  Number of I/O tensors: {num_io_tensors}")

    for i in range(num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_mode = engine.get_tensor_mode(tensor_name)

        print(f"  Tensor {i}: {tensor_name}")
        print(f"    Shape: {tensor_shape}")
        print(f"    Type: {tensor_dtype}")
        print(f"    Mode: {tensor_mode}")

    print("\nEngine build complete!")


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engine for Depth Anything V2')

    parser.add_argument(
        '--onnx',
        type=str,
        required=True,
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save TensorRT engine'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='fp16',
        choices=['fp32', 'fp16'],
        help='Precision mode'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4096,
        help='Max workspace size in MB'
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        default=1,
        help='Maximum batch size'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Build engine
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        precision=args.precision,
        workspace=args.workspace,
        max_batch_size=args.max_batch_size,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
