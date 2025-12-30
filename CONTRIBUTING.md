# Contributing

## Scope

This repository contains the TensorRT + C++/CUDA implementation and Python helper tools (ONNX export, engine build, validation) for Depth Anything V2.

## Development setup

- Install system deps (CUDA, TensorRT, OpenCV, CMake).
- Install Python deps:

```bash
pip3 install -r requirements.txt
```

## Build

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Testing / sanity checks

- Export ONNX + build engine: use the commands in `README.md`.
- Compare PyTorch vs TensorRT:

```bash
python3 python/compare_outputs.py --help
```

## Pull requests

- Keep PRs focused and minimal.
- Avoid committing generated artifacts (TensorRT engines, ONNX models, `build/`, results under `comparison_results/`).
- Include benchmark output and device details when changing performance-critical code.
