#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Validation of the attention pipeline on a larger, random problem:
//   scores = softmax( (Q Keyᵀ) / sqrt(dim_head) )
//   O      = scores * V
// We reuse the EXACT kernels from the headers and compare against a CPU reference.

// Problem dimensions
#define M 64    // queries
#define K 128   // head dimension (inner dim of Q Keyᵀ)
#define N 48    // keys / values
#define P 32    // value dimension (columns of V and O)

#define dim_head K       // scale = 1/sqrt(K)  (define before including attention.cuh)
#define SOFTMAX_BLOCK 128 // power-of-two block for the row-softmax reduction

#include "attention.cuh"
#include "softmax.cuh"

static void cpu_reference(const float *Q, const float *Key, const float *V,
                          float *scores, float *O) {
    const float scale = 1.0f / sqrtf((float)dim_head);
    for (int r = 0; r < M; r++) {
        float row[N];
        for (int c = 0; c < N; c++) {
            float s = 0.0f;
            for (int l = 0; l < K; l++) s += Q[r * K + l] * Key[c * K + l];
            row[c] = s * scale;
        }
        // numerically stable softmax over the N keys
        float mx = -INFINITY;
        for (int c = 0; c < N; c++) mx = fmaxf(mx, row[c]);
        float sum = 0.0f;
        for (int c = 0; c < N; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
        for (int c = 0; c < N; c++) { row[c] /= sum; scores[r * N + c] = row[c]; }
        // O = scores * V
        for (int cc = 0; cc < P; cc++) {
            float o = 0.0f;
            for (int l = 0; l < N; l++) o += row[l] * V[l * P + cc];
            O[r * P + cc] = o;
        }
    }
}

// Print the top-left corner of a matrix so large outputs stay readable
static void print_corner(const char *name, const float *a, int rows, int cols) {
    int pr = rows < 4 ? rows : 4;
    int pc = cols < 6 ? cols : 6;
    printf("%s (%dx%d, showing %dx%d corner):\n", name, rows, cols, pr, pc);
    for (int r = 0; r < pr; r++) {
        printf("  ");
        for (int c = 0; c < pc; c++) printf("% .4f ", a[r * cols + c]);
        printf("%s\n", pc < cols ? "..." : "");
    }
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int main() {
    srand(1234); // fixed seed => reproducible

    float *h_Q   = (float*)malloc(M * K * sizeof(float));
    float *h_Key = (float*)malloc(N * K * sizeof(float));
    float *h_V   = (float*)malloc(N * P * sizeof(float));
    for (int i = 0; i < M * K; i++) h_Q[i]   = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * K; i++) h_Key[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < N * P; i++) h_V[i]   = (float)rand() / RAND_MAX - 0.5f;

    float *ref_scores = (float*)malloc(M * N * sizeof(float));
    float *ref_O      = (float*)malloc(M * P * sizeof(float));
    cpu_reference(h_Q, h_Key, h_V, ref_scores, ref_O);

    // Device buffers
    float *d_Q, *d_Key, *d_V, *d_C, *d_O;
    cudaMalloc(&d_Q,   M * K * sizeof(float));
    cudaMalloc(&d_Key, N * K * sizeof(float));
    cudaMalloc(&d_V,   N * P * sizeof(float));
    cudaMalloc(&d_C,   M * N * sizeof(float));
    cudaMalloc(&d_O,   M * P * sizeof(float));
    cudaMemcpy(d_Q,   h_Q,   M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Key, h_Key, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,   h_V,   N * P * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the real kernels
    dim3 block(16, 16);
    dim3 gridScores((N + 15) / 16, (M + 15) / 16);
    dim3 gridOut((P + 15) / 16, (M + 15) / 16);

    attention<<<gridScores, block>>>(d_Q, d_Key, d_C, M, K, N);   // scaled QKᵀ -> C
    softmax<<<M, SOFTMAX_BLOCK>>>(d_C, d_C, M, N);                // row-wise softmax, in place
    attention_values<<<gridOut, block>>>(d_C, d_V, d_O, M, N, P); // O = scores * V
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    float *gpu_scores = (float*)malloc(M * N * sizeof(float));
    float *gpu_O      = (float*)malloc(M * P * sizeof(float));
    cudaMemcpy(gpu_scores, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_O,      d_O, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Attention scores  softmax(QKᵀ/sqrt(%d)) ===\n", dim_head);
    print_corner("GPU scores", gpu_scores, M, N);
    print_corner("CPU scores", ref_scores, M, N);
    printf("\n=== Output  O = scores * V ===\n");
    print_corner("GPU O", gpu_O, M, P);
    print_corner("CPU O", ref_O, M, P);

    // Sanity: every score row must sum to ~1
    float worst_rowsum_err = 0.0f;
    for (int r = 0; r < M; r++) {
        float s = 0.0f;
        for (int c = 0; c < N; c++) s += gpu_scores[r * N + c];
        worst_rowsum_err = fmaxf(worst_rowsum_err, fabsf(s - 1.0f));
    }

    float err_scores = max_abs_diff(gpu_scores, ref_scores, M * N);
    float err_O      = max_abs_diff(gpu_O,      ref_O,      M * P);
    printf("\nmax|scores diff| = %e\nmax|O diff|      = %e\n", err_scores, err_O);
    printf("worst |row sum - 1| = %e\n", worst_rowsum_err);
    printf("%s\n", (err_scores < 1e-5f && err_O < 1e-5f) ? "RESULT: PASS" : "RESULT: FAIL");

    free(h_Q); free(h_Key); free(h_V);
    free(ref_scores); free(ref_O); free(gpu_scores); free(gpu_O);
    cudaFree(d_Q); cudaFree(d_Key); cudaFree(d_V); cudaFree(d_C); cudaFree(d_O);
    return 0;
}
