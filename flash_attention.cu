#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

#define X 128  // Number of rows in A and C
#define Y 256   // Number of columns in A and rows in B
#define Z 64  // Number of columns in B and C
#define BLOCK_SIZE 16
#define SOFTMAX_BLOCK 256   // 1D block size for the row-wise softmax pass
#define dim_head  64
#define TILE_SIZE 16

// Softmax kernel shared from softmax.cuh (must define SOFTMAX_BLOCK / dim_head first).
#include "online_softmax.cuh"

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

// Flash attention implementation
__global__ void flashattention(float *Q, float *K, float *V, float *O, int rowQ, int colQ, int head) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    
    extern __shared__ float shared_mem[];

}


int main() {
    float *h_Q, *h_K, *h_V, *h_O;
    float *d_Q, *d_K, *d_V, *d_O;
    int size_Q = X * Y * sizeof(float);
    int size_K = Y * Z * sizeof(float);
    int size_V = X * Z * sizeof(float);
    int size_O = X * Z * sizeof(float);

    // Allocate host memory
    h_Q = (float*)malloc(size_Q);
    h_K = (float*)malloc(size_K);
    h_V = (float*)malloc(size_V);
    h_O = (float*)malloc(size_O);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_Q, X, Y);
    init_matrix(h_K, Y, Z);
    init_matrix(h_V, X, Z);

    // Allocate device memory
    cudaMalloc(&d_Q, size_Q);
    cudaMalloc(&d_K, size_K);
    cudaMalloc(&d_V, size_V);
    cudaMalloc(&d_O, size_O);

    // Copy data to device
    cudaMemcpy(d_Q, h_Q, size_Q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((Z + BLOCK_SIZE - 1) / BLOCK_SIZE, (X + BLOCK_SIZE - 1) / BLOCK_SIZE);

    double gpu_total_time = 0.0;
    double start_time = get_time();
    //attention<<<gridDim, blockDim>>>(d_Q, d_K, d_V, X, Y, Z);   // scaled QKᵀ -> C
    softmax<<<X, SOFTMAX_BLOCK>>>(d_O, d_O, X, Z);              // row-wise softmax, in place
    //attention_values<<<gridDim, blockDim>>>(d_C, d_V, d_O, X, Z, Z); // O = weights * V
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time = end_time - start_time;

    printf("Flash Attention time: %f microseconds\n", (gpu_total_time * 1e6f));

    // Free memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return 0;
}