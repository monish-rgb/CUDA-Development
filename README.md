# CUDA Development

A personal playground for learning CUDA C/C++ programming — device introspection,
and CPU-vs-GPU benchmarks for classic parallel workloads. Built and tested on
Windows 11 with an NVIDIA GPU and CUDA 13.0.

## Contents

| File | Description |
|------|-------------|
| [`deviceQuery.cu`](deviceQuery.cu) | Prints the full CUDA device configuration (SMs, clocks, memory, threading limits) plus derived theoretical peaks — FP32/FP64 TFLOPS and memory bandwidth. |
| [`gpu_config.py`](gpu_config.py) | The same device query in pure Python, talking to the NVIDIA driver via `ctypes` (`nvcuda.dll` / `libcuda.so`). No PyTorch or extra packages required. |
| [`vec_add.cu`](vec_add.cu) | Vector addition (10M elements) benchmarking a CPU loop against a CUDA kernel, with warm-up runs and result verification. |
| [`malmul.cu`](malmul.cu) | Naive matrix multiplication `(M×K) @ (K×N)` on CPU vs. a 2D-grid CUDA kernel, with timing and speedup reporting. |
| [`requirements.txt`](requirements.txt) | Python dependencies (PyTorch + NumPy) for the CUDA 13.0 wheel index. |

## Requirements

- NVIDIA GPU with an installed driver
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 13.0 (provides `nvcc`)
- A host C++ compiler (MSVC on Windows / GCC or Clang on Linux)
- Python 3.9+ (optional, for `gpu_config.py` and the PyTorch tooling)

## Building & Running (CUDA C/C++)

Compile any `.cu` file with `nvcc`, **always passing `-arch=sm_86`** (see the
note below on why this is required here):

```bash
nvcc -arch=sm_86 deviceQuery.cu -o deviceQuery
nvcc -arch=sm_86 vec_add.cu     -o vec_add
nvcc -arch=sm_86 malmul.cu      -o matmul
```

Then run the resulting executable, e.g.:

```bash
./deviceQuery      # or deviceQuery.exe on Windows
./vec_add
./matmul
```

The benchmarks perform warm-up iterations, time both the CPU and GPU
implementations over multiple runs, and print the average times and speedup.

### Why `-arch=sm_86` is required

The GPU used here is an **NVIDIA RTX 3050 Ti (compute capability 8.6)**. Without
`-arch`, `nvcc` targets a newer virtual architecture and ships the kernel as
**PTX** that gets JIT-compiled by the driver at load time. This driver rejects
it with:

```
Kernel error: the provided PTX was compiled with an unsupported toolchain.
```

When that happens the kernel **silently never runs** — you get an impossibly
fast GPU time (single-digit µs), a huge `Max difference`, and `Results are
incorrect`, even though the code is correct.

Passing `-arch=sm_86` makes `nvcc` compile directly to native SASS for the GPU's
exact compute capability (8.6), so there is no JIT step and no toolchain
mismatch. Adjust the number for a different GPU (e.g. `sm_75` for Turing,
`sm_89` for Ada). Always error-check kernel launches with `cudaGetLastError()`
so failures like this surface instead of masquerading as wrong results.

## Python Tooling

Query the GPU with no third-party dependencies:

```bash
python gpu_config.py
```

Or set up the PyTorch environment:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130
```

## Reference Materials

- [infatoshi/cuda-course](https://github.com/infatoshi/cuda-course) — the CUDA course these
  exercises follow along with.
- [CUDA Series: Streams and Synchronization](https://medium.com/@dmitrijtichonov/cuda-series-streams-and-synchronization-873a3d6c22f4)
  — Dmitrij Tichonov's write-up on CUDA streams and synchronization.
