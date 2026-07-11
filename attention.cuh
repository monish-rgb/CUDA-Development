#pragma once
#include <math.h>

// Scaled dot-product attention kernels, shared across files.
// dim_head sets the softmax scale 1/sqrt(dim_head); define it before including
// this header to override the default.
#ifndef dim_head
#define dim_head 64
#endif

// C = (Q * Key^T) * scale        Q: m x k    Key: n x k (row = one key)    C: m x n
// Each thread computes one score C[row][col] = scale * dot(Q[row], Key[col]).
static __global__ void attention(float *Q, float *Key, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float scale = 1.0f / sqrtf((float)dim_head);

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += Q[row * k + l] * Key[col * k + l]; // Key[col] is the col-th key row => implicit transpose
        }
        C[row * n + col] = sum * scale;
    }
}

// O = C * V                      C: m x n    V: n x p    O: m x p
// Each thread computes one O[row][col] as a dot product over the n values.
static __global__ void attention_values(float *C, float *V, float *O, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int l = 0; l < n; l++) {
            sum += C[row * n + l] * V[l * p + col];
        }
        O[row * p + col] = sum;
    }
}
