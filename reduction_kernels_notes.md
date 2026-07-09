# CUDA Sum Reduction Kernels — Notes

Notes explaining two sum-reduction kernels from PMPP (Programming Massively
Parallel Processors), traced with concrete arrays.

---

## 1. Naive Sum Reduction (`SimpleSumReductionKernel`)

```cuda
__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}
```

### What `i` is

Each thread handles **two elements**, so thread `t` starts at index `i = 2t`.
With `blockDim.x = 4`:

| threadIdx.x | i = 2*threadIdx.x |
|-------------|-------------------|
| 0 | 0 |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |

So 4 threads cover 8 array elements (indices 0–7).

### Trace with example array

```
index:  0   1   2   3   4   5   6   7
value:  3   1   7   2   5   8   4   6
```
Total sum = 36.

The loop is `stride = 1, 2, 4`. A thread acts only when `threadIdx.x % stride == 0`.

**Stride = 1** (all threads act) — `input[i] += input[i+1]`:
- t0: input[0] += input[1] → 3+1 = 4
- t1: input[2] += input[3] → 7+2 = 9
- t2: input[4] += input[5] → 5+8 = 13
- t3: input[6] += input[7] → 4+6 = 10

```
value:  4   1   9   2   13  8   10  6
```

**Stride = 2** (t0, t2) — `input[i] += input[i+2]`:
- t0: input[0] += input[2] → 4+9 = 13
- t2: input[4] += input[6] → 13+10 = 23

**Stride = 4** (t0) — `input[i] += input[i+4]`:
- t0: input[0] += input[4] → 13+23 = 36 ✓

Final: thread 0 writes `*output = input[0] = 36`.

### Why it is inefficient
- **Warp divergence:** `threadIdx.x % stride == 0` leaves scattered active threads;
  half go idle at stride 2, only one works at stride 4.
- **Uncoalesced access:** `input[i + stride]` with `i = 2*threadIdx.x` is strided.
- Operates directly on **global** memory.
- Overwrites the input array in place.

---

## 2. Segmented Sum Reduction (`SegmentedSumReductionKernel`)

```cuda
__global__ void SegmentedSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```

### The index variables

With `BLOCK_DIM = blockDim.x = 4`, each block processes `2 * blockDim.x = 8`
elements. Using 2 blocks → 16 elements total.

| blockIdx.x | segment | thread t | i = segment + t | i + BLOCK_DIM |
|-----------|---------|----------|-----------------|---------------|
| 0 | 0 | 0 | 0 | 4 |
| 0 | 0 | 1 | 1 | 5 |
| 0 | 0 | 2 | 2 | 6 |
| 0 | 0 | 3 | 3 | 7 |
| 1 | 8 | 0 | 8 | 12 |
| 1 | 8 | 1 | 9 | 13 |
| 1 | 8 | 2 | 10 | 14 |
| 1 | 8 | 3 | 11 | 15 |

`i` is now **contiguous** (0,1,2,3 within a block) → neighboring threads read
neighboring memory → **coalesced** access.

### Trace with example array (16 elements)

```
index:  0  1  2  3  4  5  6  7   8  9 10 11 12 13 14 15
value:  3  1  7  2  5  8  4  6   9  2  1  5  7  3  6  4
```
Total sum = 73.

**Line 6** — `input_s[t] = input[i] + input[i + BLOCK_DIM]`:

Block 0 (elements 0–7):
- t0: input[0]+input[4] = 3+5 = 8
- t1: input[1]+input[5] = 1+8 = 9
- t2: input[2]+input[6] = 7+4 = 11
- t3: input[3]+input[7] = 2+6 = 8

`input_s = [8, 9, 11, 8]`

Block 1 (elements 8–15):
- t0: input[8]+input[12] = 9+7 = 16
- t1: input[9]+input[13] = 2+3 = 5
- t2: input[10]+input[14] = 1+6 = 7
- t3: input[11]+input[15] = 5+4 = 9

`input_s = [16, 5, 7, 9]`

**The loop** — `stride = blockDim.x/2 = 2, then 1`. Only `t < stride` work, so
active threads stay contiguous (minimal divergence).

Block 0 `[8, 9, 11, 8]`:
- stride = 2: t0: 8+11=19, t1: 9+8=17 → `[19, 17, 11, 8]`
- stride = 1: t0: 19+17=36 → `[36, ...]`

Block 1 `[16, 5, 7, 9]`:
- stride = 2: t0: 16+7=23, t1: 5+9=14 → `[23, 14, ...]`
- stride = 1: t0: 23+14=37 → `[37, ...]`

**Line 14** — each block's thread 0 does `atomicAdd(output, input_s[0])`:
- Block 0 adds 36
- Block 1 adds 37
- `*output = 36 + 37 = 73` ✓

---

## Why `segment = 2 * blockDim.x * blockIdx.x`

`segment` = starting index of the array chunk this block owns.

- **`2 * blockDim.x`** = number of elements each block consumes. Each thread reads
  two elements in line 6 (`input[i]` and `input[i + BLOCK_DIM]`), so
  `blockDim.x threads × 2 = 2 * blockDim.x` elements.
- **`blockIdx.x`** = which block's turn it is, placing chunks back-to-back.

| blockIdx.x | segment = 2·4·blockIdx.x | elements owned |
|-----------|--------------------------|----------------|
| 0 | 0 | 0–7 |
| 1 | 8 | 8–15 |
| 2 | 16 | 16–23 |
| 3 | 24 | 24–31 |

Without the `2`, chunks would be spaced `blockDim.x` apart while each spans
`2*blockDim.x` → overlaps and skipped elements.

### How we know a block consumes 2 * blockDim.x elements

Line 6 is the **only** place global `input` is read. Each thread touches exactly
two indices: `input[t]` (first half) and `input[t + blockDim.x]` (second half).
Listing them (BLOCK_DIM = 4) gives indices 0–7 = 8 = 2×4 distinct elements, with
no gaps or overlap. It is a consequence of the read pattern, not an assumption.

---

## Can BLOCK_DIM be any value (1, 2, 5, …)?

**Must be a power of 2** (1, 2, 4, 8, 16, 32, …). Non-powers like 5 silently
produce wrong answers because the tree loop halves the stride with integer
division and drops elements.

- **blockDim.x = 1** ✓ — loop never runs (`1/2 = 0`), single slot already holds the sum.
- **blockDim.x = 2** ✓ — stride 1 folds the two slots. Correct.
- **blockDim.x = 5** ✗ — `input_s = [A,B,C,D,E]`:
  - stride = 5/2 = 2: t0: A+C, t1: B+D. **E (index 4) never touched.**
  - stride = 1: t0: (A+C)+(B+D).
  - stride = 0: stop.
  - Result = A+B+C+D — **E silently dropped.** Wrong answer, no crash.

### Hardware constraints (even among powers of 2)
- Must be ≥ 1 and ≤ 1024 (max threads per block).
- Ideally a multiple of 32 (warp size) for efficiency → in practice 128, 256, 512.

### Handling arbitrary array lengths
Add a bounds guard and pad missing loads with 0:

```cuda
float a = (i      < N) ? input[i]             : 0.0f;
float b = (i + BD < N) ? input[i + BLOCK_DIM] : 0.0f;
input_s[t] = a + b;
```
This fixes array length, but the tree loop still needs `blockDim.x` to be a power of 2.

---

## What `atomicAdd` does after each segment's sum

After the loop, each block has reduced its chunk to one number in `input_s[0]`
(Block 0 → 36, Block 1 → 37, …). These partial sums must be combined into one
global `output`. `atomicAdd(output, input_s[0])` does this safely.

### Why not `*output += input_s[0]`?
`+=` is three steps (read, add, write) and blocks run concurrently → race condition:

```
output starts at 0
Block 0                     Block 1
read output   -> 0
                            read output   -> 0
add 36        -> 36
                            add 37        -> 37
write output  -> 36
                            write output  -> 37   (overwrites 36!)
Final output = 37   WRONG (should be 73)
```

### What atomicAdd guarantees
Makes the read-modify-write one indivisible operation; hardware serializes
competing atomics on the same address:

```
Block 0: atomicAdd(output, 36) -> 36
Block 1: atomicAdd(output, 37) -> 73
Final output = 73   CORRECT
```

Each block's value is added exactly once. (It also returns the old value, ignored here.)

### Two caveats
1. **Zero `output` first** — `cudaMemset(output, 0, ...)` before launch, since every
   block adds onto it.
2. **Order is arbitrary** — blocks finish in any order, so float sums can differ in
   the last bit run-to-run (float addition isn't associative).

### Big picture: two-level reduction
```
Level 1 (shared memory, per block): threads tree-reduce their chunk -> input_s[0]
Level 2 (global memory, across blocks): atomicAdd folds every input_s[0] into output
```
atomicAdd is the cheap, correct glue that merges block results without a second
kernel launch.
