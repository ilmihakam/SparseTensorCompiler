# Fairness Contract for Sparse Kernel Benchmarks

This document defines the reproducibility and fairness rules used by the sparse-output benchmark runners for:
- `spadd`
- `spelmul`
- `spgemm`
- `sddmm`

## 1) Input Canonicalization

- All `.mtx` inputs are canonicalized once into `benchmarks/matrices/canonical/`.
- Canonicalization rules:
  - Expand `symmetric` and `skew-symmetric`.
  - Sort coordinates deterministically.
  - Merge duplicate coordinates by summing values.
  - Emit `%%MatrixMarket matrix coordinate real general`.
- Both our generated binaries and TACO drivers consume the same canonical files.

## 2) Timed Region Definition

- Timings report kernel compute only.
- Outside timed region:
  - Input loading/parsing.
  - One-time compile/assemble.
  - One-time output-structure assembly for sparse outputs.
  - Per-iteration output value resets (`vals` clear).
- Inside timed region:
  - Compute path only (`compute` / generated kernel compute).

## 3) Output Representation

- Output tensors are declared sparse (`CSR` or `CSC`) for all four kernels.
- Kernel output strategies:
  - `spadd`, `spelmul`, `sddmm`: fixed sparse structure assembled once, compute fills only values.
  - `spgemm`: sparse structure assembled once, compute fills only values.
- Dense-output behavior remains unchanged for unrelated existing tests and flows.

## 4) Format Constraints

- Each benchmark binary uses one sparse format per run (`csr` or `csc`).
- For two-sparse kernels in a single run:
  - `A` and `B` use the same chosen format.
  - Output `C` uses that same format.
- Benchmarks run both formats separately (`csr`, `csc`).
- Exception (explicitly tagged in output and CSV):
  - `spgemm_taco` for requested `csc` uses `Kernel mode: csc_safe_fallback_via_csr`.
  - These rows are recorded with `config=taco_csc_safe_fallback_via_csr`.

## 5) Pair Selection and Reproducibility

- Two-sparse pair lists are auto-selected deterministically from canonical metadata:
  - `spadd`/`spelmul`: `rows, cols` must match.
  - `spgemm`: `A.cols == B.rows`.
- Candidate order is lexicographic `(matrix_a, matrix_b)`; first `N` pairs are used.
- Actual selected pairs are persisted to:
  - `benchmarks/results/csv/spadd_pairs_used.csv`
  - `benchmarks/results/csv/spelmul_pairs_used.csv`
  - `benchmarks/results/csv/spgemm_pairs_used.csv`

## 6) Defaults, Seeds, and Toolchain

- No manual schedule tuning is applied to either implementation:
  - Our compiler uses existing default config set (`baseline`, `interchange_only`, `block_only`, `i_then_b`, `b_then_i`, `i_b_i`).
  - TACO drivers use default compilation/assembly/compute behavior.
- Optional benchmarking extensions (opt-in via runner flags):
  - Config sweeps add additional compiler configurations (for example block-size sweeps).
  - Multi-trial runs (`--trials N`) execute the same configuration multiple times and select the median-by-`avg_time_ms` trial.
- Deterministic dense operand initialization uses seed `42`.
- Generated C kernels are built with:
  - `gcc -O2 -march=native -std=c11`
- TACO drivers are built with:
  - `-O2 -march=native`

## 7) CSV Schemas

- Two-sparse kernels (`spadd`, `spelmul`, `spgemm`) emit:
  - `matrix_a,matrix_b,format,kernel,impl,config,rows_a,cols_a,nnz_a,rows_b,cols_b,nnz_b,iterations,total_time_ms,avg_time_ms,min_time_ms,max_time_ms,stddev_ms,variance_pct,max_error`
- `sddmm` emits:
  - `matrix_name,format,kernel,impl,config,rows,cols,nnz,K,iterations,total_time_ms,avg_time_ms,min_time_ms,max_time_ms,stddev_ms,variance_pct,max_error`

Additive columns may be present (do not break existing columns):
- `config_flags`: the exact compiler flags used for that config (or `taco` for TACO rows)
- `trials`: number of process-level trials executed for this row
- `trial_selected`: 0-based index of the selected median-by-`avg_time_ms` trial
- `trial_avg_time_ms_{median,min,max,stddev,variance_pct}`: across-trial statistics of the per-trial `avg_time_ms`
- `trial_max_error_max`: maximum `max_error` observed across trials
