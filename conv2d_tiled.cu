#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

#define M 256         // Image is M x M (rows = cols)
#define r 2            // Filter radius
#define K (2 * r + 1)  // Filter is K x K (5 x 5 for r = 2)
#define BLOCK_SIZE 32  // 32 threads per block in x and y dimensions(32,32)
#define TILE_WIDTH 32  // Tile width for shared memory optimization

__constant__ float F[K * K];

// CPU reference for conv2d (same-size output, zero padding).
// Filter is passed explicitly here since __constant__ F is device-only.
void conv2d_cpu(float *A, float *Filter, float *C, int width, int height) {
    for (int outRow = 0; outRow < height; outRow++) {
        for (int outCol = 0; outCol < width; outCol++) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += Filter[fRow * K + fCol] * A[inRow * width + inCol];
                    }
                }
            }
            C[outRow * width + outCol] = Pvalue;
        }
    }
}

// CUDA kernel for tiled conv2d using shared memory and constant memory for the filter
__global__ void conv2d(float *A, float *C, int width, int height) {
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    __shared__ float Ns[TILE_WIDTH][TILE_WIDTH]; // Shared memory for input tile
    if (row < height && col < width) {
        Ns[threadIdx.y][threadIdx.x] = A[row * width + col];
    } else {
        Ns[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
                int sRow = (int)threadIdx.y - r + fRow;
                int sCol = (int)threadIdx.x - r + fCol;
                if (sCol >= 0 && sCol < TILE_WIDTH &&
                    sRow >= 0 && sRow < TILE_WIDTH) {

                    Pvalue += F[fRow * K + fCol] * Ns[sRow][sCol];
                } else {
                    if (row - r + fRow >= 0 &&
                        row - r + fRow < height &&
                        col - r + fCol >= 0 &&
                        col - r + fCol < width) {

                        Pvalue += F[fRow * K + fCol] *
                                  A[(row - r + fRow) * width +
                                    col - r + fCol];
                    }
                }
            }
        }
        C[row * width + col] = Pvalue;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

int main() {
    float *h_A, *h_filter, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_filter, *d_C;
    int size_A = M * M * sizeof(float);
    int size_filter = K * K * sizeof(float);
    int size_C = M * M * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_filter = (float*)malloc(size_filter);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, M);
    init_matrix(h_filter, K, K);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_filter, size_filter);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_filter, h_filter, size_filter, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, h_filter, size_filter); // Copy filter to constant memory

    // Define grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH); // 16,16 threads per block
    dim3 gridDim((M + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH); // 2,2 blocks in x and y dimensions

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        conv2d<<<gridDim, blockDim>>>(d_A, d_C, M, M);
        cudaDeviceSynchronize();
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
    }

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        conv2d<<<gridDim, blockDim>>>(d_A, d_C, M, M);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("constant memory GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));

    // Verify results against the CPU reference
    conv2d_cpu(h_A, h_filter, h_C_cpu, M, M);
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    float max_diff = 0.0f;
    for (int i = 0; i < M * M; i++) {
        float diff = fabs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4) correct = false;
    }
    printf("Max difference: %e\n", max_diff);
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_A);
    free(h_filter);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_filter);
    cudaFree(d_C);

    return 0;
}
