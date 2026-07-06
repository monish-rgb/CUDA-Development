# Tiled conv2d — Shared Memory, Halo & Ghost Cells

A walkthrough of the **cached tiled 2D convolution** kernel (constant-memory
filter + shared-memory input tile), and how it cuts global-memory traffic.
Companion to [`conv2d_tiled.cu`](conv2d_tiled.cu) /
[`conv2d_constantmem.cu`](conv2d_constantmem.cu).

Config used throughout: `TILE_DIM = 16`, `FILTER_RADIUS = 2` (so the filter is
5×5).

---

## The kernel being explained

```c
__global__ void convolution_cached_tiled_2D_const_mem_kernel(
        float *N, float *P, int width, int height) {
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;

    // --- Phase 1: cooperative load of this block's tile ---
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row < height && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    else
        N_s[threadIdx.y][threadIdx.x] = 0.0;      // off-image → ghost → 0
    __syncthreads();                              // wait for all loads

    // --- Phase 2: compute, reading shared where possible ---
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                if (threadIdx.x-FILTER_RADIUS+fCol >= 0 &&
                    threadIdx.x-FILTER_RADIUS+fCol < TILE_DIM &&
                    threadIdx.y-FILTER_RADIUS+fRow >= 0 &&
                    threadIdx.y-FILTER_RADIUS+fRow < TILE_DIM) {
                    // input is inside MY tile → SHARED memory (fast)
                    Pvalue += F[fRow][fCol] * N_s[threadIdx.y+fRow][threadIdx.x+fCol];
                } else {
                    // input is a HALO cell (neighbor block) → GLOBAL memory
                    if (row-FILTER_RADIUS+fRow >= 0 && row-FILTER_RADIUS+fRow < height &&
                        col-FILTER_RADIUS+fCol >= 0 && col-FILTER_RADIUS+fCol < width) {
                        Pvalue += F[fRow][fCol] *
                            N[(row-FILTER_RADIUS+fRow)*width + col-FILTER_RADIUS+fCol];
                    }
                    // else: off-image ghost → contributes 0
                }
            }
        }
        P[row*width+col] = Pvalue;
    }
}
```

> Note: `__syncthreads()` needs the leading underscores (some textbook scans
> show `syncthreads()` — that will not compile). `F` is the filter in
> `__constant__` memory.

---

## Why shared memory helps: neighbors reuse the same inputs

In the **naive** kernel every thread reads its full `(2r+1)²` window straight
from **global memory** `N[]`. Adjacent output pixels overlap heavily — for a 5×5
filter, two side-by-side outputs share 20 of their 25 inputs. So the same global
element is fetched up to `(2r+1)²` times. Global memory (DRAM) is slow.

The fix: **load each input element from global memory once, into fast on-chip
shared memory, then let every thread that needs it read from there.**

| Kernel | Global reads per output pixel |
|--------|-------------------------------|
| Naive | `(2r+1)²` — e.g. **25** for 5×5 |
| Tiled (this) | ≈ **1** for interior threads; halo threads add a few |

Each element loaded into shared memory is reused by up to `(2r+1)²` threads but
paid for with **one** DRAM fetch → roughly a **25× reduction** in global reads
for a 5×5 filter.

---

## `N_s` is one block's tile — NOT the whole image

```c
__shared__ float N_s[TILE_DIM][TILE_DIM];   // 16×16, same size as the block
```

- `N_s` is `TILE_DIM × TILE_DIM`, **not** `width × height`.
- Each block allocates its **own private** `N_s` and loads only the image slice
  its threads cover.
- Shared memory is tiny (~48–100 KB per SM) — a full image can't fit. That is
  precisely *why* we tile.

### Each block loads a different, non-overlapping tile

The load index is offset by `blockIdx`, so same `threadIdx` in different blocks
maps to different image pixels:

```c
int col = blockIdx.x*TILE_DIM + threadIdx.x;
int row = blockIdx.y*TILE_DIM + threadIdx.y;
N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
```

| Block | Image region loaded (16×16 tile) |
|-------|----------------------------------|
| `(0,0)` | rows 0–15, cols 0–15 |
| `(1,0)` | rows 0–15, cols 16–31 |
| `(0,1)` | rows 16–31, cols 0–15 |
| `(1,1)` | rows 16–31, cols 16–31 |

```
        Full image N (lives in GLOBAL memory)
   ┌─────────────┬─────────────┬─────────────┐
   │  Block(0,0) │  Block(1,0) │  Block(2,0) │  each block copies ONLY
   │   → its N_s │   → its N_s │   → its N_s │  its own tile into its
   ├─────────────┼─────────────┼─────────────┤  own private N_s
   │  Block(0,1) │  Block(1,1) │  Block(2,1) │
   │   → its N_s │   → its N_s │   → its N_s │
   └─────────────┴─────────────┴─────────────┘
   N_s (per block) = just ONE 16×16 tile — never the whole image
```

### The whole tile *is* in `N_s` — one thread, one cell

The block has `16 × 16 = 256` threads, each loading exactly one pixel, so the
256 threads fill all 256 cells of `N_s` with no gaps:

| Thread `(tx,ty)` | writes `N_s[ty][tx]` | from pixel |
|------------------|----------------------|------------|
| (0,0)   | `N_s[0][0]`   | `N[0][0]`   |
| (15,0)  | `N_s[0][15]`  | `N[0][15]`  |
| (0,1)   | `N_s[1][0]`   | `N[1][0]`   |
| (15,15) | `N_s[15][15]` | `N[15][15]` |

`N_s` holds **exactly** the tile — nothing more. It does **not** include the
halo. (In the *other* "halo-loaded" variant, `N_s` is enlarged to
`(TILE_DIM + 2·FILTER_RADIUS)²` = 20×20 so the halo *is* included and no global
fallback is needed. This cached variant keeps the plain 16×16 tile.)

---

## `N_s` is "constant" once loaded (by convention)

- **Phase 1 (before the barrier):** each thread **writes** its one cell.
- **`__syncthreads()`:** waits until every write is done.
- **Phase 2 (after the barrier):** `N_s` appears only on the right-hand side —
  it is **read-only** for the rest of the kernel; nobody overwrites it.

So after load + barrier, `N_s` is a **frozen snapshot** of the block's tile.
Two caveats:

1. This is a **usage convention, not hardware**. Shared memory is read-write;
   the kernel simply chooses not to write again. (Constant memory `F` is
   hardware-enforced read-only.)
2. "Unchanging once loaded" is **per block, per launch**. Each block has its own
   `N_s` with a different tile, freed when the block ends.

---

## Halo vs Ghost cells

Both are inputs a thread needs that lie **outside its own tile**, but they are
different things.

| | **Halo cell** | **Ghost cell** |
|---|---|---|
| Does the data exist? | **Yes** — real pixel | **No** — off the image entirely |
| Whose region? | A **neighboring block** | Nobody's — outside `N` |
| Caused by | Tile boundary (block edge) | Image boundary |
| Value used | The actual pixel value | **0** (zero padding) |
| In this code | `else` branch → read from **global** `N[]` | inner bounds-check fails → contributes **0** |
| Exists in a single-block launch? | No (no neighbors) | **Yes** (image edges still there) |

```
                Full image N (● = real pixels)
   ┌───────────────────────────────────────┐
   │ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●   │
   │ ●  ┌───────────────┐ ● ┌──────────┐    │
   │ ●  │ This block's  │←halo→│ neighbor │  │  halo = neighbor's
   │ ●  │  tile (N_s)   │ ● │  block    │    │         real pixels
   │ ●  └───────────────┘ ● └──────────┘    │
   │ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●   │
   └───────────────────────────────────────┘
 ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
   ○ ○ ○   ghost cells: outside the image, treated as 0
 └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

A halo cell can *itself* be a ghost at a corner (the neighbor region is also
off-image) — which is why the `else` branch **still** bounds-checks before
reading `N[]`.

---

## The decision each thread makes per filter tap

For every tap the thread asks: *"is the input I need inside this block's tile?"*

```
input inside [0, TILE_DIM)  ──► YES ──► read from N_s      (SHARED,  fast)
                            └─► NO  ──► it's outside this block (halo)
                                         │
                                         ├─ inside the image? ──► read N[]  (GLOBAL, slower)
                                         └─ off the image?    ──► ghost → add 0
```

- **Interior threads** (e.g. `tx=8, ty=8`): entire 5×5 window fits in `N_s` →
  **all reads shared, zero global access.**
- **Edge threads** (e.g. `tx=0, ty=0`): some taps fall outside the tile → those
  come from **global `N[]`** (halo), and off-image ones become **ghosts (0)**.

---

## One-line summary

> **Phase 1** loads each block's own `TILE_DIM×TILE_DIM` tile into shared `N_s`
> once (one thread per cell), turning `(2r+1)²` redundant global reads per pixel
> into ~1. After `__syncthreads()`, `N_s` is a frozen, read-only snapshot.
> **Phase 2** reads from `N_s` for the common interior case, and only falls back
> to global `N[]` for **halo** cells (real neighbor data outside the tile);
> off-image **ghost** cells contribute 0.
