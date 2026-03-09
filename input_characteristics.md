
Characteristics
Matrix Dimension
NNZ Number
Sparsity Density
(Measure by nnz/row or nnz/col)
Average nnz/row, nnz/column
Variance/Irregularity of nnz
Matrix Bandwidth and
profile
NNZ Clustering and Block Structure
Manhattan Distance/Locality Metrics
Spatial Locality of column indices within rows
Temporal Reuse distance of x entries
Inter-row similarity
Irregularity and Randomness of Sparse Pattern


Below is a concrete quantification recipe for each characteristic, phrased so you can implement them directly on CSR/CSC. I’ll assume a CSR matrix with:

- `m, n` = rows, cols  
- `rowptr[0..m]`, `colind[0..nnz-1]`  

You can mirror everything on CSC if needed.

***

## Matrix dimension

- Scalars:  
  - `m_rows = m`  
  - `n_cols = n`  
- Useful derived scalars:  
  - `aspect = (double)m / n`  
  - `log_m = log2(m)`, `log_n = log2(n)`

***

## NNZ number and densities

- `nnz = rowptr[m]`  
- Overall density:  
  - `density = (double)nnz / (m * (double)n)`  
- Average nnz per row/column:  
  - `avg_nnz_row = (double)nnz / m`  
  - `avg_nnz_col = (double)nnz / n`  

To get per-row counts:

```c
int *row_nnz = malloc(m * sizeof(int));
for (int i = 0; i < m; i++)
  row_nnz[i] = rowptr[i+1] - rowptr[i];
```

To get per-column counts for CSR, either build CSC once or accumulate:

```c
int *col_nnz = calloc(n, sizeof(int));
for (int p = 0; p < nnz; p++)
  col_nnz[colind[p]]++;
```

***

## Irregularity / variance of nnz

Rows:

```c
double mu_r = (double)nnz / m;
double var_r = 0.0;
for (int i = 0; i < m; i++) {
  double d = row_nnz[i] - mu_r;
  var_r += d * d;
}
var_r /= m;
double std_r = sqrt(var_r);
double cv_r = std_r / (mu_r + 1e-9);   // coefficient of variation
```

Similarly for columns using `col_nnz`:

- `mu_c, var_c, std_c, cv_c`

You can log both variance and CV as separate features.

***

## Matrix bandwidth and simple profile

For each nnz at `(i, j)` where `i` is row index and `j = colind[p]`:

```c
int max_band = 0;
int *j_min = malloc(m * sizeof(int));
int *j_max = malloc(m * sizeof(int));
for (int i = 0; i < m; i++) {
  j_min[i] = INT_MAX;
  j_max[i] = -1;
}

for (int i = 0; i < m; i++) {
  for (int p = rowptr[i]; p < rowptr[i+1]; p++) {
    int j = colind[p];
    int bw = abs(i - j);
    if (bw > max_band) max_band = bw;
    if (j < j_min[i]) j_min[i] = j;
    if (j > j_max[i]) j_max[i] = j;
  }
}
```

Features:

- `bandwidth = max_band`  
- Per-row span (only rows with nnz):

```c
double sum_span = 0.0;
int nonempty_rows = 0;
for (int i = 0; i < m; i++) {
  if (j_max[i] >= 0) {
    sum_span += (j_max[i] - j_min[i] + 1);
    nonempty_rows++;
  }
}
double avg_span = (nonempty_rows ? sum_span / nonempty_rows : 0.0);
```

You can also store `min_span`, `max_span`, quantiles if desired.

***

## NNZ clustering / block structure

### 1D clustering within rows

For each row with nnz indices `j0 < j1 < ... < j_{k-1}`:

```c
long total_runs = 0;
long total_run_len = 0;

for (int i = 0; i < m; i++) {
  int start = rowptr[i], end = rowptr[i+1];
  int k = end - start;
  if (k == 0) continue;

  int runs = 1;
  int current_run_len = 1;
  total_run_len++; // first element

  for (int t = start+1; t < end; t++) {
    if (colind[t] == colind[t-1] + 1) {
      current_run_len++;
      total_run_len++;
    } else {
      runs++;
      current_run_len = 1;
      total_run_len++;
    }
  }
  total_runs += runs;
}
double avg_runs_per_row = (double)total_runs / m;
double avg_run_len = (double)total_run_len / total_runs;
```

Also track:

- `adjacent_fraction = (# of gaps with size 1) / (# total gaps)` where gaps are `colind[t] - colind[t-1]`.

### 2D blockiness (BCSR tendency)

Choose block sizes `Br`, `Bc` (e.g., 4, 8). Build an occupancy bitmap or a hashed set of blocks:

```c
// approximate: count nnz per block (pseudocode, use hash map if sparse)
std::unordered_map<uint64_t,int> block_nnz;

for (int i = 0; i < m; i++) {
  for (int p = rowptr[i]; p < rowptr[i+1]; p++) {
    int j = colind[p];
    int br = i / Br;
    int bc = j / Bc;
    uint64_t key = ((uint64_t)br << 32) | (uint32_t)bc;
    block_nnz[key]++;
  }
}

int occupied_blocks = block_nnz.size();
double avg_nnz_per_block = (double)nnz / occupied_blocks;
double block_density = (double)(occupied_blocks * Br * Bc) / (m * (double)n);
```

You can compute these for a few `(Br,Bc)` and treat each as a separate feature.

***

## Manhattan / spatial locality metrics

### Within-row gaps

We already have the gaps for clustering; define:

```c
long gap_count = 0;
long64_t sum_gap = 0;
long gaps_eq1 = 0;

for (int i = 0; i < m; i++) {
  int start = rowptr[i], end = rowptr[i+1];
  for (int t = start+1; t < end; t++) {
    int g = colind[t] - colind[t-1];
    gap_count++;
    sum_gap += g;
    if (g == 1) gaps_eq1++;
  }
}
double avg_gap = gap_count ? (double)sum_gap / gap_count : 0.0;
double frac_adjacent = gap_count ? (double)gaps_eq1 / gap_count : 0.0;
```

***

## Temporal reuse distance of x entries (approximate)

Simulate LRU for column indices as if doing SpMV in row-major order. Use small sets modeling L1/L2 capacities in “number of distinct columns” or “cache lines” (e.g., 64, 256, 1024).

Simplified approach for each capacity `C`:

```c
// Use e.g. std::list + unordered_map or a fixed-size LRU structure
LRUCache cache(C);
long hits = 0, misses = 0;

for (int i = 0; i < m; i++) {
  for (int p = rowptr[i]; p < rowptr[i+1]; p++) {
    int j = colind[p];
    if (cache.contains(j)) hits++;
    else { misses++; cache.insert(j); }
  }
}
double hit_rate = (double)hits / (hits + misses);
```

Features:

- `hit_rate_L1`, `hit_rate_L2`, etc.  

If you want coarse reuse distance, you can instead bucket recency (e.g., last 32, 33–256, >256) in the LRU and keep counters for which bucket it was in when hit.

***

## Inter-row similarity

Use sampled Jaccard similarity on row column-sets.

1. For each row, its columns are `colind[rowptr[i]..rowptr[i+1]-1]` sorted.
2. Sample K random distinct row pairs `(i, k)` where rows are nonempty.
3. For each pair, compute:

```c
int a_start = rowptr[i], a_end = rowptr[i+1];
int b_start = rowptr[k], b_end = rowptr[k+1];

int p = a_start, q = b_start;
int inter = 0, uni = 0;
while (p < a_end && q < b_end) {
  int ja = colind[p], jb = colind[q];
  if (ja == jb) { inter++; uni++; p++; q++; }
  else if (ja < jb) { uni++; p++; }
  else { uni++; q++; }
}
uni += (a_end - p) + (b_end - q);
double jacc = (uni ? (double)inter / uni : 0.0);
```

Accumulate:

- `mean_jaccard`, maybe `p90_jaccard`.  

Sampling: choose e.g. `K = min(1000, number_of_nonempty_rows^2)` randomly to keep cost bounded.

***

## Irregularity / randomness of sparse pattern

### Degree irregularity (rows/cols)

Already captured by `cv_r`, `cv_c`.

### Column entropy

Given per-column counts `col_nnz[j]`:

```c
double H = 0.0;
for (int j = 0; j < n; j++) {
  if (col_nnz[j] == 0) continue;
  double p = (double)col_nnz[j] / nnz;
  H -= p * log(p);   // natural log
}
double H_max = log(n);
double normalized_entropy = (H_max > 0 ? H / H_max : 0.0);
```

Low entropy → a few columns dominate (high reuse); high entropy → uniform.

### Orderability to banded

Compute original bandwidth `bandwidth_orig` (above). Then apply a cheap heuristic reordering (e.g., Reverse Cuthill–McKee from any library, or just sort rows by `j_min[i]`) and recompute `bandwidth_reordered`. Define:

- `bandwidth_reduction_ratio = (double)bandwidth_orig / max(1, bandwidth_reordered)`

Higher ratio means “more structure can be exposed by reordering”.

***

## Spatial locality across rows (entry matches)

From locality models for SpMV: count “entry matches” (two rows sharing a column index). [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X00000753)

Approximate via sampling:

1. Keep `col_nnz[j]` from before.  
2. Compute:

```c
long long pair_matches = 0;
for (int j = 0; j < n; j++) {
  long long cj = col_nnz[j];
  pair_matches += cj * (cj - 1) / 2;  // number of row pairs sharing this column
}
```

Normalize:

- `avg_entry_matches_per_row_pair = 2 * pair_matches / (double)(R * (R-1))` where `R` is no. of nonempty rows.  

This is a cheap scalar proxy for temporal reuse of `x[j]` across rows. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0167739X00000753)

***

If you want, I can help you turn this into a `struct Features` and a single pass/sampled-pass C++ implementation over your CSR that outputs a flat feature vector for ML / auto-scheduling.