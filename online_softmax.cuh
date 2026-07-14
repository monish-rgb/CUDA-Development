#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 32768
#define BLOCK_SIZE 256

// CUDA kernel for softmax
__global__ void softmax(float *a, float *o, int rows, int cols) {
int row = blockDim.x * blockIdx.x + threadIdx.x;

    float x_max = -INFINITY;
    float norm = 0.0f;

    // pass 1
    for (int col = 0; col < cols; col++) {
        int i = row * cols + col;
        float curr = a[i];
        if (curr > x_max) {
            // correct the global norm here
            norm = norm * expf(x_max - curr);
            x_max = curr;
        }
        norm += expf(curr - x_max);
    }
    // pass 2
    for (int col = 0; col < cols; col++) {
        int i = row * cols + col;
        o[i] = expf(a[i] - x_max) / norm;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time (portable, uses a monotonic clock)
double get_time() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

int main() {
    float *h_a, *h_o;
    float *d_a, *d_o;
    size_t size = (size_t)ROWS * COLS * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_o = (float*)malloc(size);

    // Initialize matrix
    srand(time(NULL));
    init_matrix(h_a, ROWS, COLS);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_o, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (ROWS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // rows = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (rows + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1024 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        softmax<<<num_blocks, BLOCK_SIZE>>>(d_a, d_o, ROWS, COLS);
        cudaDeviceSynchronize();
    }

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        softmax<<<num_blocks, BLOCK_SIZE>>>(d_a, d_o, ROWS, COLS);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);

    // Free memory
    free(h_a);
    free(h_o);
    cudaFree(d_a);
    cudaFree(d_o);

    return 0;
}