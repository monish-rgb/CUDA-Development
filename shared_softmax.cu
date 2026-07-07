#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 32768
#define BLOCK_SIZE 256

// Simple CUDA error-checking macro
#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

// CUDA kernel for softmax: one block per row, threads cooperate on that row.
__global__ void softmax(float *a, float *o, int rows, int cols) {
    __shared__ float smem[BLOCK_SIZE];   // shared memory lives inside the kernel

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // edge condition (we don't process further)
    if (row >= rows) return;

    float* input_row  = a + row * cols;
    float* output_row = o + row * cols;
    float local_max  = -INFINITY;
    float local_norm = 0.0f;

    // Phase 1: per-thread online pass over this thread's slice of the row
    // (grid-stride => coalesced global-memory reads)
    for (int i = tid; i < cols; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);  // rescale running sum to new max
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }

    // Phase 2: tree reduction to find the row-wide maximum
    smem[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float global_max = smem[0];
    __syncthreads();

    // Phase 3: rescale each thread's local_norm to the GLOBAL max, then sum-reduce
    local_norm *= expf(local_max - global_max);
    smem[tid] = local_norm;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];   // sum, not max
        }
        __syncthreads();
    }
    float global_norm = smem[0];
    __syncthreads();

    // Phase 4: write the normalized output for this thread's slice
    for (int i = tid; i < cols; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - global_max) / global_norm;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// CPU reference softmax (numerically stable, row-wise) for verification
void cpu_softmax(const float *a, float *o, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *in = a + (size_t)r * cols;
        float *out = o + (size_t)r * cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; c++) m = fmaxf(m, in[c]);
        float norm = 0.0f;
        for (int c = 0; c < cols; c++) norm += expf(in[c] - m);
        for (int c = 0; c < cols; c++) out[c] = expf(in[c] - m) / norm;
    }
}

// Function to measure execution time (portable, uses a monotonic clock)
double get_time() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

int main() {
    float *h_a, *h_o, *h_ref;
    float *d_a, *d_o;
    size_t size = (size_t)ROWS * COLS * sizeof(float);

    // Allocate host memory
    h_a   = (float*)malloc(size);
    h_o   = (float*)malloc(size);
    h_ref = (float*)malloc(size);

    // Initialize matrix
    srand(time(NULL));
    init_matrix(h_a, ROWS, COLS);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_o, size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    // One block per row
    int num_blocks = ROWS;

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        softmax<<<num_blocks, BLOCK_SIZE>>>(d_a, d_o, ROWS, COLS);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        softmax<<<num_blocks, BLOCK_SIZE>>>(d_a, d_o, ROWS, COLS);
        CHECK_CUDA(cudaDeviceSynchronize());
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;
    printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);

    // Copy GPU result back and verify against CPU reference
    CHECK_CUDA(cudaMemcpy(h_o, d_o, size, cudaMemcpyDeviceToHost));

    printf("Verifying against CPU reference...\n");
    cpu_softmax(h_a, h_ref, ROWS, COLS);

    double max_abs_err = 0.0;
    for (size_t i = 0; i < (size_t)ROWS * COLS; i++) {
        double err = fabs((double)h_o[i] - (double)h_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
    }
    // Also sanity-check that a few rows sum to ~1
    double row0_sum = 0.0;
    for (int c = 0; c < COLS; c++) row0_sum += h_o[c];

    printf("Max abs error vs CPU: %e\n", max_abs_err);
    printf("Row 0 sum (should be ~1.0): %f\n", row0_sum);
    printf("%s\n", (max_abs_err < 1e-5) ? "RESULT: PASS" : "RESULT: FAIL");

    // Free memory
    free(h_a);
    free(h_o);
    free(h_ref);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_o));

    return 0;
}
