# 2D Blocking Analysis Pipeline

This document describes the analysis stage for 2D blocking in the benchmarking pipeline.

It explains:

- what inputs the analysis consumes
- how the heuristic signals are computed
- what the output CSV files contain
- how to run and interpret the analysis

## Purpose

The goal of this stage is to relate measured 2D-blocking performance to the structure of the sparse input matrix.

For each benchmark row that uses a 2D blocking config, the analysis:

1. extracts the 2D tile size from the benchmark config name
2. matches the row against its `baseline` benchmark row
3. computes `speedup_vs_baseline`
4. overlays a tile grid on the canonical sparse matrix
5. computes per-tile sparse-structure metrics
6. classifies each non-empty tile as useful or not useful
7. aggregates those tile-level observations into matrix-level heuristic coefficients

The main output is a dataset that joins benchmark performance and tile-quality signals.

## Supported kernels

The current analysis covers:

- `spmm`
- `sddmm`

These are the kernels that currently have 2D blocking support in the benchmark pipeline.

## Inputs

The analysis script is:

- `benchmarks/scripts/analyze_2d_blocking.py`

It expects:

- benchmark CSVs from the unified runners:
  - `benchmarks/results/csv/benchmark_spmm.csv`
  - `benchmarks/results/csv/benchmark_sddmm.csv`
- canonical matrices:
  - `benchmarks/matrices/suitesparse/canonical/*.mtx`

The benchmark CSVs must contain 2D rows whose `config` fields look like:

- `block2d_b16x32`
- `all2d_I_THEN_B_b32x64`
- `all2d_B_THEN_I_b32x64`
- `all2d_I_B_I_b32x64`

If the CSVs do not contain any `block2d_*` or `all2d_*` configs, the analysis will stop with an explicit error.

## How benchmark rows are selected

The script reads the benchmark CSVs and filters to rows where:

- `kernel` is `spmm` or `sddmm`
- `impl` matches the requested implementation list
- `config` matches a 2D blocking pattern

For each selected row, the script builds a baseline lookup key:

- `spmm`: `(kernel, impl, format, matrix_name, N)`
- `sddmm`: `(kernel, impl, format, matrix_name, K)`

The speedup is:

```text
speedup_vs_baseline = baseline_avg_time_ms / avg_time_ms
```

If either timing is zero or missing, the speedup is recorded as `nan`.

## How tile sizes are extracted

The config name determines the geometric tile size used by the analysis.

Examples:

```text
block2d_b16x32           -> tile_i = 16, tile_j = 32, order = ""
all2d_I_THEN_B_b32x64   -> tile_i = 32, tile_j = 64, order = I_THEN_B
```

These analysis tile sizes are applied directly to the sparse matrix coordinate space.

## Sparse tile model

For a matrix of shape `m x n` and tile size `(Ti, Tj)`, the matrix is divided into a geometric grid:

```text
num_row_tiles = ceil(m / Ti)
num_col_tiles = ceil(n / Tj)
```

Tile `(p, q)` covers:

```text
rows [p*Ti, min((p+1)*Ti, m))
cols [q*Tj, min((q+1)*Tj, n))
```

Only tiles containing at least one nonzero are written to the per-tile CSV.

## Per-tile metrics

For each non-empty tile, the script computes:

- `tile_nnz`
  - number of nonzeros in the tile
- `density`
  - `tile_nnz / tile_area`
- `span`
  - `max_col_in_tile - min_col_in_tile + 1`
- `span_frac`
  - `span / tile_width`
- `row_cv`
  - coefficient of variation of per-row nnz counts within the tile
- `runs_avg`
  - average number of contiguous column runs per row in the tile

These metrics are intended to reflect whether the tile looks clustered and structured enough to benefit from blocking.

## Useful tile definition

A tile is classified as useful if it passes all enabled metric checks.

Default checks:

- `tile_nnz >= nnz_min`
- `density >= density_min`
- `span_frac <= span_frac_max`
- `row_cv <= cv_max`
- `runs_avg >= runs_min`

Default threshold values:

- `nnz_min = 8`
- `density_min = 0.05`
- `span_frac_max = 0.5`
- `cv_max = 2.0`
- `runs_min = 1.5`

Each check can be enabled or disabled independently from the CLI.

## Why both non-empty and empty-tile signals are kept

Two different questions matter:

1. how good are the tiles that actually contain sparse work?
2. how fragmented is the matrix over the chosen tile grid?

These are captured separately:

- `R_tile_nonempty`
  - fraction of non-empty tiles that are useful
- `empty_tile_frac`
  - fraction of the full geometric tile grid that is empty

This matters because empty tiles do not necessarily create direct sparse-work overhead in the current implementation, but a high empty-tile fraction can still indicate that the matrix is fragmented and globally less favorable for blocking.

## CSR and CSC handling

The analysis computes tile metrics in two modes:

- `as_is`
- `transpose`

The script always emits both.

Selected mode:

- `csr` benchmark rows use `as_is`
- `csc` benchmark rows use `transpose`

This gives a format-aware view while preserving both variants for inspection.

## Heuristic coefficients

The main matrix-level signals produced by the analysis are:

- `r_tile_nonempty`
  - `useful_nonempty_tiles / nonempty_tiles`
- `r_tile_all`
  - `useful_tiles / total_tiles_in_grid`
- `empty_tile_frac`
  - `empty_tiles / total_tiles_in_grid`
- `good_to_bad_ratio`
  - `useful_nonempty_tiles / non_useful_nonempty_tiles`
- `r_nnz_useful`
  - `nnz in useful tiles / total nnz`

Current interpretation:

- `r_tile_nonempty` measures how favorable the occupied tiles are
- `empty_tile_frac` measures how fragmented the matrix is under the chosen tile size
- `good_to_bad_ratio` is an alternate way to view tile balance
- `r_nnz_useful` shows how much actual sparse work falls inside useful tiles

In practice, `r_tile_nonempty` and `empty_tile_frac` should be studied together.

## Output files

The analysis produces three CSV files.

### 1. Dataset CSV

Default:

- `benchmarks/results/analysis/2d_blocking_dataset.csv`

This file contains one row per analyzed 2D benchmark row.

It includes:

- original benchmark row fields
- parsed 2D config fields:
  - `config_kind`
  - `order`
  - `tile_i`
  - `tile_j`
- performance fields:
  - `baseline_avg_time_ms`
  - `speedup_vs_baseline`
- threshold metadata:
  - `threshold_profile_id`
- aggregate tile statistics for both modes:
  - `*_as_is`
  - `*_transpose`
- selected-format copies:
  - `*_selected`

Examples of aggregate columns:

- `tiles_nonempty_*`
- `tiles_empty_*`
- `tiles_good_*`
- `tiles_bad_*`
- `empty_tile_frac_*`
- `r_tile_nonempty_*`
- `r_tile_all_*`
- `r_nnz_useful_*`
- `good_to_bad_ratio_*`
- `tile_nnz_mean_*`, `tile_nnz_p50_*`, `tile_nnz_max_*`
- `density_mean_*`, `density_p50_*`, `density_max_*`
- `span_frac_mean_*`, `span_frac_p50_*`, `span_frac_max_*`
- `row_cv_mean_*`, `row_cv_p50_*`, `row_cv_max_*`
- `runs_avg_mean_*`, `runs_avg_p50_*`, `runs_avg_max_*`

### 2. Per-tile CSV

Default:

- `benchmarks/results/analysis/2d_blocking_tiles.csv`

This file contains one row per non-empty tile.

Columns include:

- `threshold_profile_id`
- `kernel`
- `matrix_name`
- `mode`
- `tile_i`
- `tile_j`
- `tile_p`
- `tile_q`
- `tile_nnz`
- `density`
- `span`
- `span_frac`
- `row_cv`
- `runs_avg`
- `is_good`
- `passes_nnz`
- `passes_density`
- `passes_span`
- `passes_cv`
- `passes_runs`

This is the main file to inspect when you want to understand why a matrix/tile size combination is or is not favorable.

### 3. Thresholds-used CSV

Default:

- `benchmarks/results/analysis/2d_blocking_thresholds_used.csv`

This file records the exact threshold profile for the run:

- `threshold_profile_id`
- `nnz_min`
- `density_min`
- `span_frac_max`
- `cv_max`
- `runs_min`
- `use_nnz`
- `use_density`
- `use_span`
- `use_cv`
- `use_runs`

## CLI usage

Minimal usage:

```bash
cd benchmarks/scripts
python3 analyze_2d_blocking.py
```

Explicit paths:

```bash
cd benchmarks/scripts
python3 analyze_2d_blocking.py \
  --spmm-csv ../results/csv/benchmark_spmm.csv \
  --sddmm-csv ../results/csv/benchmark_sddmm.csv \
  --canonical-dir ../matrices/suitesparse/canonical \
  --output ../results/analysis/2d_blocking_dataset.csv \
  --tiles-output ../results/analysis/2d_blocking_tiles.csv \
  --thresholds-output ../results/analysis/2d_blocking_thresholds_used.csv
```

Threshold tuning example:

```bash
cd benchmarks/scripts
python3 analyze_2d_blocking.py \
  --nnz-min 4 \
  --density-min 0.02 \
  --span-frac-max 0.75 \
  --cv-max 3.0 \
  --runs-min 1.0
```

Disable selected criteria:

```bash
cd benchmarks/scripts
python3 analyze_2d_blocking.py \
  --no-use-span \
  --no-use-runs
```

## Recommended workflow

1. Run `spmm` and `sddmm` benchmarks with 2D configs enabled.
2. Run `analyze_2d_blocking.py` on the resulting CSVs.
3. Inspect `2d_blocking_dataset.csv` to compare:
   - `speedup_vs_baseline`
   - `r_tile_nonempty_selected`
   - `empty_tile_frac_selected`
   - `good_to_bad_ratio_selected`
4. Inspect `2d_blocking_tiles.csv` for matrices/tile sizes that look surprising.
5. Tune thresholds and rerun analysis until the heuristic separates helpful and unhelpful 2D-blocking cases more clearly.

## Interpreting the outputs

Common patterns:

- high `r_tile_nonempty_selected` and low `empty_tile_frac_selected`
  - the tile grid looks structurally favorable for blocking
- high `r_tile_nonempty_selected` and high `empty_tile_frac_selected`
  - occupied tiles look good, but the matrix is globally fragmented
- low `r_tile_nonempty_selected`
  - most occupied tiles do not satisfy the current usefulness rule
- high `good_to_bad_ratio_selected`
  - useful occupied tiles dominate non-useful occupied tiles

These outputs are intended for heuristic derivation, threshold fitting, and debugging.

## Limitations

- The heuristic is threshold-based and hand-designed.
- It is not yet a fitted predictive model.
- Very small benchmark runs can produce `avg_time_ms = 0`, which causes `speedup_vs_baseline` to become `nan`.
- The current analysis focuses on sparse tile structure, not a full cache simulation.

## Suggested next step

The next natural extension is an offline fitting script that searches over threshold combinations and scores them against observed `speedup_vs_baseline`, so the usefulness heuristic is tuned from data rather than manually selected.
