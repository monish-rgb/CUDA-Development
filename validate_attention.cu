#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Small, hand-checkable validation of the attention pipeline:
//   scores = softmax( (Q Keyᵀ) / sqrt(dim_head) )
//   O      = scores * V
// We reuse the EXACT kernels from the headers and compare against a CPU reference.

#define dim_head 4         // scale = 1/sqrt(4) = 0.5  (define before including attention.cuh)
#define SOFTMAX_BLOCK 32   // power-of-two block for the row-softmax reduction

#include "attention.cuh"
#include "softmax.cuh"

// Problem dimensions (tiny on purpose)
#define M 2   // queries
#define K 4   // head dimension (inner dim of Q Keyᵀ)
#define N 3   // keys / values
#define P 2   // value dimension (columns of V and O)

static void cpu_reference(const float *Q, const float *Key, const float *V,
                          float *scores, float *O) {
    const float scale = 1.0f / sqrtf((float)dim_head);
    for (int r = 0; r < M; r++) {
        // raw scaled scores for this query row
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

static void print_matrix(const char *name, const float *a, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int r = 0; r < rows; r++) {
        printf("  ");
        for (int c = 0; c < cols; c++) printf("% .4f ", a[r * cols + c]);
        printf("\n");
    }
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int main() {
    // Fixed inputs (row-major). Q: MxK, Key: NxK (one key per row), V: NxP.
    float h_Q[M * K]   = { 1, 0, 1, 0,
                           0, 1, 0, 1 };
    float h_Key[N * K] = { 1, 0, 1, 0,
                           0, 1, 0, 1,
                           1, 1, 1, 1 };
    float h_V[N * P]   = { 1, 0,
                           0, 1,
                           1, 1 };

    // CPU reference
    float ref_scores[M * N], ref_O[M * P];
    cpu_reference(h_Q, h_Key, h_V, ref_scores, ref_O);

    // Device buffers
    float *d_Q, *d_Key, *d_V, *d_C, *d_O;
    cudaMalloc(&d_Q,   sizeof(h_Q));
    cudaMalloc(&d_Key, sizeof(h_Key));
    cudaMalloc(&d_V,   sizeof(h_V));
    cudaMalloc(&d_C,   M * N * sizeof(float));
    cudaMalloc(&d_O,   M * P * sizeof(float));
    cudaMemcpy(d_Q,   h_Q,   sizeof(h_Q),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_Key, h_Key, sizeof(h_Key), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V,   h_V,   sizeof(h_V),   cudaMemcpyHostToDevice);

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

    // Copy back
    float gpu_scores[M * N], gpu_O[M * P];
    cudaMemcpy(gpu_scores, d_C, sizeof(gpu_scores), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_O,      d_O, sizeof(gpu_O),      cudaMemcpyDeviceToHost);

    // Report
    printf("=== Attention scores  softmax(QKᵀ/sqrt(%d)) ===\n", dim_head);
    print_matrix("GPU scores", gpu_scores, M, N);
    print_matrix("CPU scores", ref_scores, M, N);
    printf("\n=== Output  O = scores * V ===\n");
    print_matrix("GPU O", gpu_O, M, P);
    print_matrix("CPU O", ref_O, M, P);

    float err_scores = max_abs_diff(gpu_scores, ref_scores, M * N);
    float err_O      = max_abs_diff(gpu_O,      ref_O,      M * P);
    printf("\nmax|scores diff| = %e\nmax|O diff|      = %e\n", err_scores, err_O);
    printf("%s\n", (err_scores < 1e-5f && err_O < 1e-5f) ? "RESULT: PASS" : "RESULT: FAIL");

    cudaFree(d_Q); cudaFree(d_Key); cudaFree(d_V); cudaFree(d_C); cudaFree(d_O);
    return 0;
}
