#pragma once
#include <math.h>

// Row-wise softmax kernel, shared across files.
// One block per row; threads cooperate over that row's `cols` columns.
// Numerically stable via the online-max trick, then tree reductions for max and sum.
// Safe to call in place (in == out): each element is read once and written once.
//
// SOFTMAX_BLOCK must equal the launch block size and be a power of two.
// Define it before including this header to override the default.
#ifndef SOFTMAX_BLOCK
#define SOFTMAX_BLOCK 256
#endif

static __global__ void softmax(float *in, float *out, int rows, int cols) {
    __shared__ float smem[SOFTMAX_BLOCK];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float *input_row  = in  + row * cols;
    float *output_row = out + row * cols;
    float local_max  = -INFINITY;
    float local_norm = 0.0f;

    // Phase 1: per-thread pass over a grid-strided slice of the row (coalesced reads)
    for (int i = tid; i < cols; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x); // rescale running sum to the new max
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }

    // Phase 2: tree reduction for the row-wide maximum
    smem[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float global_max = smem[0];
    __syncthreads();

    // Phase 3: rescale each thread's partial sum to the global max, then sum-reduce
    local_norm *= expf(local_max - global_max);
    smem[tid] = local_norm;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float global_norm = smem[0];
    __syncthreads();

    // Phase 4: write the normalized probabilities for this thread's slice
    for (int i = tid; i < cols; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - global_max) / global_norm;
    }
}
