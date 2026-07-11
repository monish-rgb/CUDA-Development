#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

#define M 256  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32
#define SOFTMAX_BLOCK 256   // 1D block size for the row-wise softmax pass
#define dim_head  64

// Row-wise softmax kernel shared from softmax.cuh (must define SOFTMAX_BLOCK first)
#include "softmax.cuh"


// CUDA kernel for Q*K^T
__global__ void attention(float *Q, float *Key, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float scale = 1.0f / sqrtf((float)dim_head); // Scale factor for attention

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += Q[row * k + l] * Key[col * k + l]; // Access Key in column-major order
        }
        C[row * n + col] = sum * scale;
    }
    // Note: The above kernel computes the attention scores by performing the matrix multiplication of Q and Key^T, and then scales the result.


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
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu, *h_V;
    float *d_A, *d_B, *d_C, *d_V;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);
    int size_V = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);
    h_V = (float*)malloc(size_V);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    init_matrix(h_V, M, N);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMalloc(&d_V, size_V);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    attention<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);   // scaled QKᵀ -> C
    softmax<<<M, SOFTMAX_BLOCK>>>(d_C, d_C, M, N);              // row-wise softmax, in place
    cudaDeviceSynchronize();

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 10; i++) {
        double start_time = get_time();
        attention<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);   // scaled QKᵀ -> C
        softmax<<<M, SOFTMAX_BLOCK>>>(d_C, d_C, M, N);              // row-wise softmax, in place
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 10.0;

    // Print results
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_V);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_V);

    return 0;
}