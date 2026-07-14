#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

// Flash Attention (forward pass)

#define B   16      // batch
#define NH  12     // heads
#define N   64     // sequence length  
#define D   64     // head dimension

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int n, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * n * d) + (by * n * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * n) + (by * n);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

static inline void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

// CPU reference: standard numerically-stable attention for one (b,h) head.
// Q,K,V are [N, D] slices; out is [N, D].
static void cpu_head(const float* Q, const float* K, const float* V,
                     float* out, float scale) {
    float* row = (float*)malloc(N * sizeof(float));
    for (int qi = 0; qi < N; qi++) {
        float mx = -INFINITY;
        for (int kj = 0; kj < N; kj++) {
            float s = 0.0f;
            for (int x = 0; x < D; x++) s += Q[qi * D + x] * K[kj * D + x];
            s *= scale;
            row[kj] = s;
            if (s > mx) mx = s;
        }
        float sum = 0.0f;
        for (int kj = 0; kj < N; kj++) { row[kj] = expf(row[kj] - mx); sum += row[kj]; }
        for (int x = 0; x < D; x++) {
            float o = 0.0f;
            for (int kj = 0; kj < N; kj++) o += (row[kj] / sum) * V[kj * D + x];
            out[qi * D + x] = o;
        }
    }
    free(row);
}

int main() {
    srand(1234);

    const int Bc = 32, Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)D);

    const int total = B * NH * N * D;   // elements in Q/K/V/O
    const int lm_total = B * NH * N;    // elements in l/m

    // Host Q,K,V in (B, NH, N, D) layout -- already split, so no permute needed.
    float* h_Q = (float*)malloc(total * sizeof(float));
    float* h_K = (float*)malloc(total * sizeof(float));
    float* h_V = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++) h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < total; i++) h_K[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < total; i++) h_V[i] = (float)rand() / RAND_MAX - 0.5f;

    // CPU reference over every (b,h) head -- timed.
    float* ref_O = (float*)malloc(total * sizeof(float));
    auto cpu_t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < B; b++)
        for (int h = 0; h < NH; h++) {
            int off = (b * NH + h) * N * D;
            cpu_head(h_Q + off, h_K + off, h_V + off, ref_O + off, softmax_scale);
        }
    auto cpu_t1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    // Device buffers.
    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaCheck(cudaMalloc(&d_Q, total * sizeof(float)), "malloc Q");
    cudaCheck(cudaMalloc(&d_K, total * sizeof(float)), "malloc K");
    cudaCheck(cudaMalloc(&d_V, total * sizeof(float)), "malloc V");
    cudaCheck(cudaMalloc(&d_O, total * sizeof(float)), "malloc O");
    cudaCheck(cudaMalloc(&d_l, lm_total * sizeof(float)), "malloc l");
    cudaCheck(cudaMalloc(&d_m, lm_total * sizeof(float)), "malloc m");

    cudaCheck(cudaMemcpy(d_Q, h_Q, total * sizeof(float), cudaMemcpyHostToDevice), "cpy Q");
    cudaCheck(cudaMemcpy(d_K, h_K, total * sizeof(float), cudaMemcpyHostToDevice), "cpy K");
    cudaCheck(cudaMemcpy(d_V, h_V, total * sizeof(float), cudaMemcpyHostToDevice), "cpy V");

    // m starts at -inf (online-softmax identity); memset can't write floats,
    // so keep a host copy to re-seed d_m before each timed run.
    float* h_m = (float*)malloc(lm_total * sizeof(float));
    for (int i = 0; i < lm_total; i++) h_m[i] = -INFINITY;

    // Resets O=0, l=0, m=-inf. Must run before every kernel launch because the
    // online softmax reads+accumulates into these buffers.
    auto reset_state = [&]() {
        cudaCheck(cudaMemset(d_O, 0, total * sizeof(float)), "memset O");
        cudaCheck(cudaMemset(d_l, 0, lm_total * sizeof(float)), "memset l");
        cudaCheck(cudaMemcpy(d_m, h_m, lm_total * sizeof(float), cudaMemcpyHostToDevice), "cpy m");
    };

    // Shared memory: Qi, Kj, Vj  (each Bc*D) + S (Bc*Br).
    const int sram_size = (3 * Bc * D * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram; cudaDeviceGetAttribute(&max_sram, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested: %d\n", max_sram, sram_size);
    if (sram_size > max_sram) { fprintf(stderr, "Not enough shared memory\n"); return 1; }

    dim3 grid_dim(B, NH);   // one block per (batch, head)
    dim3 block_dim(Bc);     // Bc threads, one per row of the tile

    cudaEvent_t ev_start, ev_stop;
    cudaCheck(cudaEventCreate(&ev_start), "event start");
    cudaCheck(cudaEventCreate(&ev_stop),  "event stop");

    // Warm-up (also the run we validate against).
    reset_state();
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        d_Q, d_K, d_V, N, D, Tc, Tr, Bc, Br, softmax_scale, d_l, d_m, d_O);
    cudaCheck(cudaGetLastError(), "launch");
    cudaCheck(cudaDeviceSynchronize(), "sync");

    // Timed runs: kernel only (buffer reset is outside the CUDA-event window).
    const int ITERS = 50;
    float gpu_ms_total = 0.0f;
    for (int it = 0; it < 25; it++) {
        reset_state();
        cudaCheck(cudaEventRecord(ev_start), "record start");
        forward_kernel<<<grid_dim, block_dim, sram_size>>>(
            d_Q, d_K, d_V, N, D, Tc, Tr, Bc, Br, softmax_scale, d_l, d_m, d_O);
        cudaCheck(cudaEventRecord(ev_stop), "record stop");
        cudaCheck(cudaEventSynchronize(ev_stop), "event sync");
        float ms; cudaCheck(cudaEventElapsedTime(&ms, ev_start, ev_stop), "elapsed");
        gpu_ms_total += ms;
    }
    double gpu_ms = gpu_ms_total / ITERS;

    float* gpu_O = (float*)malloc(total * sizeof(float));
    cudaCheck(cudaMemcpy(gpu_O, d_O, total * sizeof(float), cudaMemcpyDeviceToHost), "cpy O back");

    // Compare.
    float max_diff = 0.0f;
    for (int i = 0; i < total; i++)
        max_diff = fmaxf(max_diff, fabsf(gpu_O[i] - ref_O[i]));

    printf("Config: B=%d NH=%d N=%d D=%d  (Tr=%d Tc=%d)\n", B, NH, N, D, Tr, Tc);
    printf("First head, O[0][:6]:\n  GPU:");
    for (int x = 0; x < 6 && x < D; x++) printf(" % .4f", gpu_O[x]);
    printf("\n  CPU:");
    for (int x = 0; x < 6 && x < D; x++) printf(" % .4f", ref_O[x]);
    printf("\n\nmax|O diff| = %e\n", max_diff);
    printf("%s\n\n", (max_diff < 1e-4f) ? "RESULT: PASS" : "RESULT: FAIL");

    // Timing summary.
    printf("=== Timing ===\n");
    printf("CPU reference : %10.3f ms  (1 pass over all %d heads)\n", cpu_ms, B * NH);
    printf("GPU kernel    : %10.3f ms  (avg of %d runs, kernel only)\n", gpu_ms, ITERS);
    printf("Speedup       : %10.2fx\n", cpu_ms / gpu_ms);

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    free(h_m);
    free(h_Q); free(h_K); free(h_V); free(ref_O); free(gpu_O);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_l); cudaFree(d_m);
    return 0;
}
