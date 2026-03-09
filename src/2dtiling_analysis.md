# Tile Metrics

## Current benchmark + analysis context

2D blocking is now implemented and benchmarked in the unified pipeline:

- `spmm`: 2D blocking tiles `i` and `j`
- `sddmm`: 2D blocking tiles `i` and `k`
- benchmark configs appear as `block2d_b16x32` and `all2d_I_THEN_B_b32x64`

The analysis pipeline for these configs needs to do two separate jobs:

1. extract the measured 2D-blocking rows from benchmark CSVs and compare them against `baseline`
2. compute per-tile sparse-structure metrics for the input matrix and aggregate them into a usefulness coefficient

The coefficient used by the pipeline is:

```text
R_tile_nonempty = (# useful non-empty tiles) / (# non-empty tiles)
```

We also keep:

- `empty_tile_frac = (# empty tiles) / (# total tiles in the geometric grid)`
- a good-to-bad ratio over non-empty tiles
- an nnz-weighted useful-work ratio

The reason to keep both `R_tile_nonempty` and `empty_tile_frac` is:

- `R_tile_nonempty` tells us how favorable the tiles with actual sparse work are
- `empty_tile_frac` tells us how fragmented the matrix is over the chosen tile grid

This separation matters because empty tiles do not necessarily incur direct sparse-work cost in the current implementation, but a very high empty-tile fraction can still indicate that blocking is structurally unlikely to help.

The definition of a "useful" tile is configurable. The default checks are:

- minimum nnz in the tile
- minimum density
- maximum span fraction (`span / tile_width`)
- maximum row irregularity (`row_cv`)
- minimum contiguous-run score (`runs`)

All of these thresholds should be exposed by the analysis script so they can be tuned without changing code.

Per-tile metrics: what they are + how to quantify
Per-tile metrics measure the quality of each 2D region of the sparse matrix to determine if it's worth "blocking" (i.e., would benefit from cache locality inside that tile).

What is a "tile"?
text
Matrix A (m x n):
┌─────────────────┐
│ ┌───┐ ┌───┐ ... │  ← row-tile p=0 (Ti=32 rows)
│ │   │ │   │     │
│ └───┘ └───┘     │
│ ┌───┐ ┌───┐ ... │  ← row-tile p=1
│ │   │ │   │     │
│ └───┘ └───┘     │
│     ...         │
└─────────────────┘

Each "box" is tile (p,q): Ti rows × Tj columns (e.g., 32×32)
5 key per-tile metrics (CSR-friendly)
For tile 
(
p
,
q
)
(p,q), count nnz that fall inside it:

2. density(T) = nnz(T) / (Ti * Tj)
   - Measures "fill-in" within tile bounds
   - Too sparse → mostly wasted loop iterations

3. span(T) = max_col_in_tile - min_col_in_tile + 1
   - Width of columns actually touched *within* tile
   - Too wide → defeats spatial locality (cache misses spread out)

4. row_cv(T): coefficient of variation of nnz per row *within tile*
   - row_cv = std(nnz_per_row_in_tile) / mean(nnz_per_row_in_tile)
   - Too irregular → load imbalance, poor prefetching

5. runs(T): avg number of contiguous j-runs per row *within tile*
   - Measures local clustering/adjacency
   - Too few → very scattered access pattern
How we quantify "good" vs "bad" tiles
A tile is "good" if it passes ALL thresholds (tunable):

text
good_tile = (
  nnz(T)        >= NNZ_min     # e.g., 8–16 nnz (amortizes overhead)
  AND density(T) >= 0.05       # 5% fill-in (not too sparse)
  AND span(T)    <= 2 * Tj     # spans ≤ 2x tile width
  AND row_cv(T)  <= 2.0        # std ≤ 2x mean (not too skewed)
  AND runs(T)    >= 1.5        # some local structure
)
Global decision metric: R (the coefficient)
text
W_good  = total nnz in good tiles across whole matrix
W_total = total nnz in matrix

R = W_good / W_total    # ∈ [0, 1]

Heuristic:
if R >= R_min → enable 2D blocking
else → use baseline
R_min values:

text
Global:      R_min ≈ 0.70  (70% of work in good tiles)
SpMM:        R_min ≈ 0.65  (slightly more tolerant)
SDDMM:       R_min

Global decision metric: R (the coefficient, continued)
text
R_min values (fitted from your benchmarks):
Global:      R_min ≈ 0.70  (70% of work in good tiles)
SpMM:        R_min ≈ 0.65  (slightly more tolerant, higher compute/nnz)
SDDMM:       R_min ≈ 0.60  (heaviest compute, tolerates worse tiles)
SpMV:        R_min ≈ 0.75  (lightest compute, needs better tiles)
SPADD:       R_min ≈ 0.85  (streaming, only if almost all tiles good)
SPELMUL:     R_min ≈ 0.70  (kernel-specific)
SpGEMM:      R_min ≈ 0.65  (if you adapt for row-wise)
Full workflow (end-to-end)
text
1. Pick candidate tile sizes: [16×16, 32×32, 64×64]

2. For each matrix + tile_size:
   a. Stream CSR → bucket nnz into tiles → compute 5 metrics/tile
   
   b. Classify each tile good/bad using thresholds
   
   c. R = (nnz in good tiles) / total_nnz
   
   d. Pick "best" tile_size for this matrix (highest R, or fixed)

3. Decision:
   if R_best >= R_min[kernel]: enable_2d_blocking(tile_size=best)
   else: baseline

4. Fitting (offline, using your speedup data):
   - Grid search thresholds + R_min to maximize:
     Precision: % of predicted-wins that actually speedup >= 1.05
     Recall: % of actual wins that are predicted
Why this works (hardware intuition)
text
"Good" tile ≈ "looks dense to cache":
- nnz(T) ≥ NNZ_min: enough work/loop to pay fixed costs
- density(T) ≥ 0.05: ~1 cache line worth of useful data/tile
- span(T) ≤ 2*Tj: fits prefetch window, good TLB reuse
- low row_cv(T): balanced work/row → good branch prediction
- runs(T) ≥ 1.5: some spatial locality → prefetcher helps

R ≥ 0.70: 70%+ of *actual work* (nnz) gets locality benefit

text
1. nnz(T): number of nonzeros in this tile
   - Too low → overhead dominates


## Computing tile-quality metrics for your heuristic

Here’s a **concrete, ready-to-implement procedure** to compute the per-tile metrics and aggregate R for a CSR matrix, which you can use to decide whether to enable 2D blocking.

### 1. Tile grid definition

Given matrix dimensions \(m \times n\), tile sizes \(T_i, T_j\) (e.g., 16, 32, 64):

```
num_row_tiles = ceil(m / T_i)
num_col_tiles = ceil(n / T_j)

# Each tile (p, q) covers:
#   rows [p*T_i .. min((p+1)*T_i, m))
#   cols [q*T_j .. min((q+1)*T_j, n))
```

### 2. Per-tile metrics (CSR-friendly) can extend to CSC as well

For each tile \((p, q)\), compute these 5 metrics by **streaming once** over the CSR structure:

```
# Pseudocode for CSR (pos[], crd[], vals[])
for each row_tile p:
  row_start = p * T_i
  row_end   = min((p+1) * T_i, m)
  
  for each row i in [row_start .. row_end):
    for each nnz position pA in [pos[i] .. pos[i+1]):
      j = crd[pA]  # column index
      
      # Find which column-tile this nnz belongs to
      q = floor(j / T_j)
      
      # Accumulate into tile (p, q)
      tiles[p][q].nnz += 1
      
      # Update min/max col in this tile for span
      tiles[p][q].min_col = min(tiles[p][q].min_col, j)
      tiles[p][q].max_col = max(tiles[p][q].max_col, j)
      
      # Optional: collect per-row nnz in tile for variance
      tiles[p][q].row_nnz_counts[i - row_start] += 1
```

**Metrics computed per tile \((p, q)\)**:

```
tile_area = T_i * T_j
tile_nnz  = tiles[p][q].nnz

density(T) = tile_nnz / tile_area

span(T)    = tiles[p][q].max_col - tiles[p][q].min_col + 1

# Irregularity: coefficient of variation of nnz per row *within tile*
row_nnz_var = variance(tiles[p][q].row_nnz_counts)
row_nnz_mean = mean(tiles[p][q].row_nnz_counts)
row_cv(T)  = sqrt(row_nnz_var) / row_nnz_mean   # CV = std/mean

# Optional: number of contiguous runs (adjacency)
#   (requires second pass or bucketing per row)
runs(T)    = average runs per row within tile
```

### 3. "Good tile" classification

A tile is **good** if it meets **all** these thresholds (tune these empirically):

```
if (tile_nnz >= NNZ_min)                    # Enough work to amortize overhead
   and (density(T) >= density_min)           # Not too sparse
   and (span(T) <= span_max * T_j)           # Not too spread out
   and (row_cv(T) <= cv_max)                 # Not too irregular within tile
   and (runs(T) >= runs_min):                # Some local structure
then good_tile = true
```

**Starting values** (rough defaults, fit to your data):
```
NNZ_min     = 8–16     # ~half a cache line of indices+values
density_min = 0.05–0.1 # 5–10% fill-in feels "dense-ish"
span_max    = 2–4      # span at most 2–4x tile width
cv_max      = 2.0      # std <= 2x mean nnz/row
runs_min    = 1.5      # avg 1.5+ runs per row (not singletons)
```

### 4. Aggregate to global R

```
W_good  = sum_{good tiles} tile_nnz
W_total = total_nnz

R = W_good / W_total
```

**Heuristic rule**:
```
if R >= R_min:  # e.g., 0.7
  enable_2d_blocking()
else:
  use_baseline()
```

### 5. Implementation cost and runtime

- **Time**: \(O(\text{nnz} + m)\)—one pass over CSR, constant work per nnz.
- **Space**: \(O(\text{num_row_tiles} \times \text{num_col_tiles})\) for tile stats (~few KB even for huge matrices).
- **Per-tile-size**: Repeat for each \(T_i \times T_j\) (16×16, 32×32, 64×64) → pick the size with best R or highest predicted speedup.

### 6. CSR-specific details can extend onto CSC as well

**Column tile membership**:
```
q = floor(j / T_j)
```
Exact, no approximation.

**Span within tile**:
```
span = max_col_in_tile - min_col_in_tile + 1
```
Directly measures “effective bandwidth” within the tile bounds.

**Row nnz variance**:
- Within each row-tile \(p\), bucket nnz by column-tile \(q\).
- For each \((p,q)\), compute CV over the \(T_i\) rows.

**Runs** (optional, more expensive):
- Per row within tile, count contiguous j-runs.
- Average over rows in tile.

### 7. Fitting thresholds from your benchmarks

With your existing data (matrices + measured blockonly speedups)

1. Compute R for each matrix across a grid of thresholds.
2. Plot: **R vs log2(block_speedup)** → look for clear separation (high R → speedup > 1.05, low R → speedup ≈ 1.0).
3. Pick thresholds where:
   - Precision: 80%+ of matrices with R ≥ R_min actually win.
   - Recall: you catch most of the big wins (>10% speedup).

This gives you **kernel-specific R_min** values (e.g., SpGEMM might tolerate R=0.6, SpMV needs R=0.8).

***
