// deviceQuery.cu — prints the full CUDA device configuration and derived
// theoretical peaks (FLOPS + memory bandwidth).
// Build: nvcc deviceQuery.cu -o deviceQuery.exe
// Run:   ./deviceQuery.exe
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int nDev = 0;
    cudaGetDeviceCount(&nDev);
    printf("CUDA devices found: %d\n", nDev);

    for (int d = 0; d < nDev; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        // CUDA 13 removed the deprecated clockRate/memoryClockRate struct
        // fields; query them via cudaDeviceGetAttribute (values in kHz).
        int clockKHz = 0, memClockKHz = 0;
        cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, d);
        cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, d);

        printf("\n=================== Device %d: %s ===================\n", d, p.name);
        printf("Compute capability        : %d.%d\n", p.major, p.minor);
        printf("SMs (multiprocessors)     : %d\n", p.multiProcessorCount);
        printf("Clock rate (GPU)          : %.0f MHz\n", clockKHz / 1000.0);
        printf("Memory clock              : %.0f MHz\n", memClockKHz / 1000.0);
        printf("Memory bus width          : %d-bit\n", p.memoryBusWidth);
        printf("Total global memory       : %.2f GB\n", p.totalGlobalMem / (1024.0*1024*1024));
        printf("L2 cache size             : %.2f MB\n", p.l2CacheSize / (1024.0*1024));

        printf("\n-- Threading limits --\n");
        printf("Warp size                 : %d\n", p.warpSize);
        printf("Max threads / block       : %d\n", p.maxThreadsPerBlock);
        printf("Max threads / SM          : %d\n", p.maxThreadsPerMultiProcessor);
        printf("Max block dim  (x,y,z)    : (%d, %d, %d)\n",
               p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("Max GRID dim   (x,y,z)    : (%d, %d, %d)\n",
               p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        printf("Registers / block         : %d\n", p.regsPerBlock);
        printf("Registers / SM            : %d\n", p.regsPerMultiprocessor);
        printf("Shared mem / block        : %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("Shared mem / SM           : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);

        // ---- Derived theoretical peaks ----
        // GA10x (compute 8.6) has 128 FP32 cores per SM.
        int coresPerSM = 128;
        double cudaCores = (double)coresPerSM * p.multiProcessorCount;
        double ghz = clockKHz / 1e6;                    // kHz -> GHz
        double fp32_tflops = cudaCores * 2.0 * ghz / 1000.0;  // 2 = FMA

        // GDDR6 is double-data-rate: effective rate = 2 x memory clock.
        double memRateGbps = 2.0 * (memClockKHz / 1e6); // GT/s effective
        double bw = memRateGbps * (p.memoryBusWidth / 8.0);    // GB/s

        printf("\n-- Derived theoretical peaks --\n");
        printf("CUDA cores (FP32)         : %.0f  (%d SM x %d)\n", cudaCores, p.multiProcessorCount, coresPerSM);
        printf("Peak FP32 (FMA)           : %.2f TFLOPS\n", fp32_tflops);
        printf("Peak FP64                 : %.2f TFLOPS (1/64 rate)\n", fp32_tflops / 64.0);
        printf("Memory bandwidth          : %.1f GB/s\n", bw);
    }
    return 0;
}
