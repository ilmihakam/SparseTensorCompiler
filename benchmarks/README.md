# SparseTensorCompiler Benchmarking

Unified benchmark runners live in `benchmarks/scripts/`:

- `benchmark_spmv.py`
- `benchmark_spmm.py`
- `benchmark_spadd.py`
- `benchmark_spelmul.py`
- `benchmark_spgemm.py`
- `benchmark_sddmm.py`

Each runner writes one primary CSV (`benchmark_<kernel>.csv`) and one failure CSV
(`benchmark_<kernel>_failed.csv`) under `benchmarks/results/csv/` by default.

## Quick Start

### 1) Setup TACO (optional baseline)
```bash
cd benchmarks/taco
# Follow README.md in this directory, then build all *_taco drivers
```

### 2) Run any kernel benchmark
```bash
cd benchmarks/scripts
python3 benchmark_spmv.py
python3 benchmark_spmm.py
python3 benchmark_spadd.py
python3 benchmark_spelmul.py
python3 benchmark_spgemm.py
python3 benchmark_sddmm.py
```

Or run all kernels in one shot:

```bash
cd benchmarks/scripts
python3 benchmark_all.py
```

By default each runner:
- canonicalizes `benchmarks/matrices/suitesparse/raw/*.mtx` into `benchmarks/matrices/suitesparse/canonical/`
- runs `csr,csc`
- runs the 6 compiler configs (`baseline`, `interchange_only`, `block_only`, `i_then_b`, `b_then_i`, `i_b_i`)
- includes TACO if the corresponding `benchmarks/taco/build/<kernel>_taco` exists

## Custom Matrix Generation

You can generate synthetic matrices with configurable characteristics
(density, irregularity, bandwidth/profile, clustering, hotspot columns, inter-row similarity, blockiness):

```bash
python3 benchmarks/scripts/generate_matrices.py \
  --spec benchmarks/matrices/generated/specs/example.json \
  --out benchmarks/matrices/generated \
  --force
```

Then benchmark generated matrices:

```bash
python3 benchmarks/scripts/benchmark_all.py \
  --matrices-dir benchmarks/matrices/generated/raw \
  --canonical-dir benchmarks/matrices/generated/canonical \
  --matrix-manifest benchmarks/matrices/generated/manifest.csv \
  --max-matrices 10
```

Benchmark SuiteSparse matrices in the parallel dataset layout:

```bash
python3 benchmarks/scripts/benchmark_all.py \
  --matrices-dir benchmarks/matrices/suitesparse/raw \
  --canonical-dir benchmarks/matrices/suitesparse/canonical \
  --max-matrices 10
```

Useful selection flags:
- `--matrix-manifest <path>`: attach generator metadata + matrix characteristics to benchmark rows
- `--matrices a,b,c`: benchmark only selected matrix stems
- `--pairs-file <path>` (pair kernels): benchmark explicit pairs (`matrix_a,matrix_b`, optional `kernel` column)

## Config Sweeps and Trials

All runners support:
- `--trials N`: run each (matrix/pair, format, config) `N` times and select the median-by-`avg_time_ms` trial
- `--sweep-block` / `--sweep-block-sizes 16,32,64`: add block-size sweep configs
- `--sweep-orders I_THEN_B,B_THEN_I,I_B_I`: opt orders for generated `all_*` sweep configs
- `--hwc-mode perf --hwc-events cycles,instructions`: wrap each selected trial with `perf stat`
- `--hwc-strict`: fail a row if requested hardware-counter events are unavailable

`benchmark_spmm.py` and `benchmark_sddmm.py` additionally support:
- `--sweep-block-2d` / `--sweep-block-2d-sizes 16x16,32x32,64x64`: add 2D-blocking sweep configs

Examples:
```bash
cd benchmarks/scripts
python3 benchmark_spmv.py --trials 3 --sweep-block
python3 benchmark_spmm.py --trials 3 --sweep-block --sweep-block-2d
python3 benchmark_all.py --trials 3 --sweep-block --sweep-block-2d
python3 benchmark_spmv.py --trials 3 --hwc-mode perf --hwc-events cycles,instructions
```

When `--hwc-mode perf` is enabled, the runners append `hwc_*` columns to the CSV output. Use only event names that your guest exposes in `perf list`; do not assume Intel-specific cache-event names on `arm64`.

### 3) View outputs
```bash
ls ../results/csv/benchmark_*.csv
```

`benchmark_all.py` forwards shared options (for example `--formats`, `--configs`, `--iterations`, `--no-taco`) and writes:
- `benchmark_<kernel>.csv`
- `benchmark_<kernel>_failed.csv`
- `<kernel>_pairs_used.csv` (for `spadd`, `spelmul`, `spgemm`)

## Deterministic Selection

- `spmv`, `spmm`, `sddmm` use lexicographically sorted matrices and select first `--max-matrices`.
- `spadd`, `spelmul`, `spgemm` use deterministic compatible pairs and select first `--max-pairs`.
- Two-sparse runners persist selected pairs to:
  - `spadd_pairs_used.csv`
  - `spelmul_pairs_used.csv`
  - `spgemm_pairs_used.csv`

## Smoke Test

Run unified smoke tests:

```bash
cd benchmarks/scripts
python3 test_unified_benchmark_pipeline.py --mode ours --kernel all
python3 test_unified_benchmark_pipeline.py --mode taco --kernel all
```

## 2D Blocking Analysis

The 2D blocking analysis pipeline for `spmm` / `sddmm` is documented in:

- `benchmarks/2D_BLOCKING_ANALYSIS.md`
