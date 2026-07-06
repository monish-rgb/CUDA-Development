# conv2d Naive Kernel — Thread & Block Flow

A walkthrough of how [`conv2d_naive.cu`](conv2d_naive.cu) maps CUDA threads and
blocks onto a 2D convolution. Config used throughout:

| Constant | Value | Meaning |
|----------|-------|---------|
| `M` | 64 | Image is `M x M` (64×64) |
| `r` | 2 | Filter radius |
| `K` | `2*r+1` = 5 | Filter is `K x K` (5×5) |
| `BLOCK_SIZE` | 32 | Threads per block, per axis |

---

## Step 1 — Host decides the launch geometry

```c
dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);   // (32, 32)
dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
             (M + BLOCK_SIZE - 1) / BLOCK_SIZE);   // ((64+31)/32, ...) = (2, 2)
```

Two 2D quantities are defined **before** the kernel runs:

| | x | y | meaning |
|---|---|---|---|
| `blockDim` | 32 | 32 | threads **per block** → 32×32 = **1024 threads/block** |
| `gridDim` | 2 | 2 | blocks **in the grid** → 2×2 = **4 blocks** |

The ceiling division `(M + BLOCK_SIZE - 1) / BLOCK_SIZE` guarantees enough blocks
to cover all 64 columns/rows even when `M` is not a clean multiple of 32.

**Total threads launched** = grid × block = `(2·32) × (2·32)` = `64 × 64` =
**4096** — one thread per output pixel, exactly.

---

## Step 2 — The launch

```c
conv2d<<<gridDim, blockDim>>>(d_A, d_filter, d_C, M, M);
```

`<<<gridDim, blockDim>>>` tells the GPU: *create 4 blocks, each with 1024
threads.* The GPU hands these blocks to its SMs (Streaming Multiprocessors);
blocks run independently and possibly in parallel. Inside each block, threads
execute in **warps of 32**.

The 4-block grid laid over the 64×64 output image:

```
        cols 0–31        cols 32–63
      ┌───────────────┬───────────────┐
rows  │  Block (0,0)  │  Block (1,0)  │
0–31  │  32×32 thr    │  32×32 thr    │
      ├───────────────┼───────────────┤
rows  │  Block (0,1)  │  Block (1,1)  │
32–63 │  32×32 thr    │  32×32 thr    │
      └───────────────┴───────────────┘
```

*(Block index is `(blockIdx.x, blockIdx.y)`.)*

---

## Step 3 — Each thread figures out which pixel it owns

Every one of the 4096 threads runs the same code but with **different built-in
index values**:

```c
int outCol = blockIdx.x * blockDim.x + threadIdx.x;
int outRow = blockIdx.y * blockDim.y + threadIdx.y;
```

Built-in variables the GPU provides to each thread:

- `threadIdx.x/.y` — position **within** its block (0–31)
- `blockIdx.x/.y` — **which** block (0–1)
- `blockDim.x/.y` — block size (32), same for all threads
- `gridDim.x/.y` — grid size (2), available but not needed here

The formula `blockIdx * blockDim + threadIdx` converts *local* coordinates into
a *global* pixel coordinate. Worked examples:

| Thread | blockIdx | threadIdx | outCol = bx·32+tx | outRow = by·32+ty | owns pixel |
|--------|----------|-----------|-------------------|-------------------|------------|
| A | (0,0) | (0,0)   | 0  | 0  | C[0][0]   |
| B | (0,0) | (5,3)   | 5  | 3  | C[3][5]   |
| C | (1,0) | (0,0)   | 32 | 0  | C[0][32]  |
| D | (1,1) | (31,31) | 63 | 63 | C[63][63] |

The 4 blocks tile the image perfectly, each thread grabbing one distinct
`(outRow, outCol)`.

---

## Step 4 — Bounds guard (the grid-overhang check)

```c
if (outRow >= height || outCol >= width) return;
```

At `M = 64` this triggers for **zero** threads (64 is exactly 2×32). But with
`M = 40`, the grid is still `ceil(40/32) = 2` blocks = 64 threads/axis, so
threads 40–63 have no valid pixel and must `return` before writing. This is what
makes the kernel safe for **any** `M`.

---

## Step 5 — Each thread does its own convolution

Every surviving thread independently computes **its one output pixel** by
looping over the 5×5 filter:

```c
float Pvalue = 0.0f;
for (int fRow = 0; fRow < 2*r+1; fRow++)          // 5 rows
    for (int fCol = 0; fCol < 2*r+1; fCol++) {    // 5 cols
        int inRow = outRow - r + fRow;            // centered window
        int inCol = outCol - r + fCol;
        if (inRow >= 0 && inRow < height &&
            inCol >= 0 && inCol < width)          // halo / zero-pad check
            Pvalue += Filter[fRow*(2*r+1)+fCol] * A[inRow*width + inCol];
    }
C[outRow * width + outCol] = Pvalue;              // write my pixel
```

All 4096 threads run this **concurrently** (subject to how many the GPU
schedules at once). There is **no communication between threads** — each reads
from `A`/`Filter` and writes one cell of `C`. That independence is exactly why
it parallelizes so well.

### The two boundary checks are different

| | Check in Step 4 (`outRow/outCol`) | Check in Step 5 (`inRow/inCol`) |
|---|---|---|
| **Guards** | the output **write** to `C` | each input **read** from `A` |
| **Protects against** | surplus grid threads (block padding) | filter window hanging off the image edge |
| **Runs** | once, at the top | every filter tap (25× for a 5×5) |
| **On failure** | `return` — skip the whole pixel | skip that one tap, keep going (= zero padding) |

---

## The whole flow in one picture

```
HOST                              DEVICE (GPU)
────                              ────────────
blockDim=(32,32) ┐
gridDim =(2,2)   ├─ <<<grid,block>>> ─► spawn 4 blocks × 1024 threads = 4096
                 ┘                        │
                                          ├─ each thread:
                                          │    outCol = blockIdx.x*32 + threadIdx.x
                                          │    outRow = blockIdx.y*32 + threadIdx.y
                                          │    if (out of image) return;          ← Step 4
                                          │    loop 5×5 filter, zero-pad edges     ← Step 5
                                          │    C[outRow][outCol] = Pvalue
cudaDeviceSynchronize() ◄─── wait ────────┘
cudaMemcpy(d_C → h_C_gpu)  copy results back
```

**Mental model:** you carve the output image into 32×32 tiles (blocks), and
inside each tile every pixel gets its own worker (thread).
`blockIdx * blockDim + threadIdx` is the address translation that tells each
worker exactly which pixel of the full image it is responsible for.
