# SuiteSparse Matrices

Directory layout:
- `raw/`: downloaded `.mtx` files from SuiteSparse
- `canonical/`: canonicalized matrices used for benchmark execution
- `manifest.csv` (optional): metadata if you want explicit dataset tracking

Benchmark against SuiteSparse matrices:

```bash
python3 benchmarks/scripts/benchmark_all.py \
  --matrices-dir benchmarks/matrices/suitesparse/raw \
  --canonical-dir benchmarks/matrices/suitesparse/canonical \
  --max-matrices 10
```

If you maintain a manifest, add:

```bash
--matrix-manifest benchmarks/matrices/suitesparse/manifest.csv
```
