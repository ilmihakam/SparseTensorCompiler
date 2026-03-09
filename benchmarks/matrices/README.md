# Test Matrices

## Sources

All matrices from SuiteSparse Matrix Collection:
https://sparse.tamu.edu/

## Dataset Layout

- `suitesparse/raw` and `suitesparse/canonical` for downloaded matrices
- `generated/raw` and `generated/canonical` for synthetic matrices

## Current Collection

### Small (<10K rows)
- **bcsstk17**: https://sparse.tamu.edu/HB/bcsstk17
  - 10974 × 10974, 428K nnz
  - Finite element, structural

- **nasa4704**: https://sparse.tamu.edu/DNVS/nasa4704
  - 4704 × 4704, 104K nnz
  - NASA structural matrix

- **bcsstm22**: https://sparse.tamu.edu/HB/bcsstm22
  - 138 × 138, 696 nnz
  - Very small, for quick debugging

### Medium (10K-100K rows)
- **cage12**: https://sparse.tamu.edu/vanHeukelum/cage12
  - 130K × 130K, 2M nnz
  - DNA electrophoresis, irregular

- **mac_econ**: https://sparse.tamu.edu/Williams/mac_econ
  - 206K × 206K, 1.3M nnz
  - Economic model, very sparse

## Download Instructions

### Option 1: Direct Download (Recommended)

```bash
cd benchmarks/matrices/suitesparse/raw

# Download bcsstk17
curl -L https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz | tar -xz
mv bcsstk17/bcsstk17.mtx ./
rm -rf bcsstk17

# Download nasa4704
curl -L https://suitesparse-collection-website.herokuapp.com/MM/DNVS/nasa4704.tar.gz | tar -xz
mv nasa4704/nasa4704.mtx ./
rm -rf nasa4704

# Download bcsstm22
curl -L https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstm22.tar.gz | tar -xz
mv bcsstm22/bcsstm22.mtx ./
rm -rf bcsstm22
```

### Option 2: Manual Download

1. Visit https://sparse.tamu.edu/
2. Search for matrix by name
3. Download the `.tar.gz` file
4. Extract the `.mtx` file
5. Place in `benchmarks/matrices/suitesparse/raw/`

## Adding New Matrices

1. Download from SuiteSparse
2. Extract .mtx file
3. Place in `benchmarks/matrices/suitesparse/raw/`
4. Update this README with matrix info

## Generated Matrices

For synthetic, configurable matrices use:

```bash
python3 benchmarks/scripts/generate_matrices.py \
  --spec benchmarks/matrices/generated/specs/example.json \
  --out benchmarks/matrices/generated \
  --force
```

This writes generated files under `benchmarks/matrices/generated/`.
