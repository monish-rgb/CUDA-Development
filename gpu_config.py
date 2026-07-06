"""gpu_config.py -- print the full CUDA device configuration and derived
theoretical peaks (FLOPS + memory bandwidth), using only the standard library.

It talks to the installed NVIDIA driver through the CUDA Driver API
(nvcuda.dll on Windows / libcuda.so on Linux) via ctypes, so it needs no
extra Python packages -- not even PyTorch. This is the Python equivalent of
the classic CUDA `deviceQuery` sample.

Run:  python gpu_config.py
"""

import ctypes
import sys

# --- CUdevice_attribute enum values we care about (from cuda.h) ---
ATTR = {
    "MAX_THREADS_PER_BLOCK": 1,
    "MAX_BLOCK_DIM_X": 2,
    "MAX_BLOCK_DIM_Y": 3,
    "MAX_BLOCK_DIM_Z": 4,
    "MAX_GRID_DIM_X": 5,
    "MAX_GRID_DIM_Y": 6,
    "MAX_GRID_DIM_Z": 7,
    "MAX_SHARED_MEMORY_PER_BLOCK": 8,
    "WARP_SIZE": 10,
    "MAX_REGISTERS_PER_BLOCK": 12,
    "CLOCK_RATE": 13,                       # kHz
    "MULTIPROCESSOR_COUNT": 16,
    "MEMORY_CLOCK_RATE": 36,                # kHz
    "GLOBAL_MEMORY_BUS_WIDTH": 37,          # bits
    "L2_CACHE_SIZE": 38,                    # bytes
    "MAX_THREADS_PER_MULTIPROCESSOR": 39,
    "COMPUTE_CAPABILITY_MAJOR": 75,
    "COMPUTE_CAPABILITY_MINOR": 76,
    "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": 81,
    "MAX_REGISTERS_PER_MULTIPROCESSOR": 82,
}


def load_driver():
    for name in ("nvcuda.dll", "libcuda.so", "libcuda.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    sys.exit("Could not load the CUDA driver (nvcuda.dll / libcuda.so). "
             "Is an NVIDIA driver installed?")


def check(cuda, code, what):
    if code != 0:
        sys.exit(f"CUDA driver call failed ({what}): error {code}")


def main():
    cuda = load_driver()

    check(cuda, cuda.cuInit(0), "cuInit")

    count = ctypes.c_int()
    check(cuda, cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    print(f"CUDA devices found: {count.value}")

    def attr(name, dev):
        val = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(val), ATTR[name], dev)
        return val.value

    for d in range(count.value):
        dev = ctypes.c_int()
        check(cuda, cuda.cuDeviceGet(ctypes.byref(dev), d), "cuDeviceGet")
        dev = dev.value

        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, dev)

        total = ctypes.c_size_t()
        # cuDeviceTotalMem is exported as the versioned symbol on modern drivers.
        fn = getattr(cuda, "cuDeviceTotalMem_v2", None) or cuda.cuDeviceTotalMem
        fn(ctypes.byref(total), dev)

        major = attr("COMPUTE_CAPABILITY_MAJOR", dev)
        minor = attr("COMPUTE_CAPABILITY_MINOR", dev)
        sms = attr("MULTIPROCESSOR_COUNT", dev)
        gpu_khz = attr("CLOCK_RATE", dev)
        mem_khz = attr("MEMORY_CLOCK_RATE", dev)
        bus = attr("GLOBAL_MEMORY_BUS_WIDTH", dev)

        print(f"\n{'='*15} Device {d}: {name.value.decode()} {'='*15}")
        print(f"Compute capability        : {major}.{minor}")
        print(f"SMs (multiprocessors)     : {sms}")
        print(f"Clock rate (GPU)          : {gpu_khz/1000:.0f} MHz")
        print(f"Memory clock              : {mem_khz/1000:.0f} MHz")
        print(f"Memory bus width          : {bus}-bit")
        print(f"Total global memory       : {total.value/1024**3:.2f} GB")
        print(f"L2 cache size             : {attr('L2_CACHE_SIZE', dev)/1024**2:.2f} MB")

        print("\n-- Threading limits --")
        print(f"Warp size                 : {attr('WARP_SIZE', dev)}")
        print(f"Max threads / block       : {attr('MAX_THREADS_PER_BLOCK', dev)}")
        print(f"Max threads / SM          : {attr('MAX_THREADS_PER_MULTIPROCESSOR', dev)}")
        print(f"Max block dim  (x,y,z)    : ({attr('MAX_BLOCK_DIM_X', dev)}, "
              f"{attr('MAX_BLOCK_DIM_Y', dev)}, {attr('MAX_BLOCK_DIM_Z', dev)})")
        print(f"Max GRID dim   (x,y,z)    : ({attr('MAX_GRID_DIM_X', dev)}, "
              f"{attr('MAX_GRID_DIM_Y', dev)}, {attr('MAX_GRID_DIM_Z', dev)})")
        print(f"Registers / block         : {attr('MAX_REGISTERS_PER_BLOCK', dev)}")
        print(f"Registers / SM            : {attr('MAX_REGISTERS_PER_MULTIPROCESSOR', dev)}")
        print(f"Shared mem / block        : {attr('MAX_SHARED_MEMORY_PER_BLOCK', dev)//1024} KB")
        print(f"Shared mem / SM           : {attr('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', dev)//1024} KB")

        # ---- Derived theoretical peaks ----
        # GA10x / Ampere consumer (compute 8.x) = 128 FP32 cores per SM.
        cores_per_sm = 128 if major == 8 else 64
        cuda_cores = cores_per_sm * sms
        ghz = gpu_khz / 1e6
        fp32_tflops = cuda_cores * 2 * ghz / 1000.0        # 2 flops per FMA
        mem_rate_gbps = 2 * (mem_khz / 1e6)                 # GDDR6 = double data rate
        bw = mem_rate_gbps * (bus / 8.0)                    # GB/s

        print("\n-- Derived theoretical peaks --")
        print(f"CUDA cores (FP32)         : {cuda_cores}  ({sms} SM x {cores_per_sm})")
        print(f"Peak FP32 (FMA)           : {fp32_tflops:.2f} TFLOPS")
        print(f"Peak FP64                 : {fp32_tflops/64:.2f} TFLOPS (1/64 rate)")
        print(f"Memory bandwidth          : {bw:.1f} GB/s")


if __name__ == "__main__":
    main()
