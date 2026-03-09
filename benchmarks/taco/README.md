# TACO Integration Setup

## Prerequisites

TACO (Tensor Algebra Compiler) must be built from source as an external dependency.

## Step 1: Clone and Build TACO

```bash
# Clone TACO repository
git clone https://github.com/tensor-compiler/taco.git ~/taco
cd ~/taco

# Build TACO
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Note**: TACO build may take 10-15 minutes depending on your machine.

## Step 2: Build TACO Drivers

```bash
cd /Users/ilmihakam/co/backend/SparseTensorCompiler/benchmarks/taco
mkdir build && cd build

# Point CMake to TACO installation
cmake .. -DTACO_DIR=~/taco/build
make
```

## Step 3: Verify Installation

```bash
# Test spmv_taco with a small matrix
./spmv_taco .../../matrices/suitesparse/raw/bcsstm22.mtx 100 csr
./spmv_taco .../../matrices/suitesparse/raw/bcsstm22.mtx 100 csc

# Sparse-output kernels
./spadd_taco .../../matrices/suitesparse/canonical/bcsstk01.mtx .../../matrices/suitesparse/canonical/bcsstk01.mtx 100 csr
./spelmul_taco .../../matrices/suitesparse/canonical/bcsstk01.mtx .../../matrices/suitesparse/canonical/bcsstk01.mtx 100 csc
./spgemm_taco .../../matrices/suitesparse/canonical/bcsstk01.mtx .../../matrices/suitesparse/canonical/bcsstk01.mtx 100 csr
./sddmm_taco .../../matrices/suitesparse/canonical/bcsstk01.mtx 64 100 csc
```

`spgemm_taco` runs `csc` requests in a correctness-safe fallback mode (`Kernel mode: csc_safe_fallback_via_csr`).
This keeps the comparison reproducible while avoiding known incorrect default CSC behavior in this TACO setup.

Expected output format (matching our compiler):
```
Matrix: .../../matrices/suitesparse/raw/bcsstm22.mtx (138 x 138, 696 nnz)
Format: csr
Warmup iterations: 1
Iterations: 100
Total time: X.XX ms
Avg time per iteration: X.XXXX ms
Max error vs reference: X.XXe-YY
Implementation: TACO
```

## Troubleshooting

### CMake Cannot Find TACO

If `find_package(TACO)` fails:

1. Check that TACO was built successfully: `ls ~/taco/build/lib`
2. Try absolute path: `cmake .. -DTACO_DIR=/absolute/path/to/taco/build`
3. Set CMAKE_PREFIX_PATH: `cmake .. -DCMAKE_PREFIX_PATH=~/taco/build`

### Linking Errors

If you get undefined reference errors:

1. Ensure TACO was built with same compiler (gcc vs clang)
2. Try rebuilding TACO with `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`

### Compilation Errors

If C++ standard errors occur:

1. Ensure your compiler supports C++17
2. Try explicit flag: `cmake .. -DCMAKE_CXX_STANDARD=17`

## Optional: Test TACO Independently

```bash
cd ~/taco/build
./bin/taco-test  # Run TACO's own tests
```

## Integration with Benchmarking

Once built, unified runners in `benchmarks/scripts/benchmark_<kernel>.py` automatically detect and use:
- `benchmarks/taco/build/spmv_taco`
- `benchmarks/taco/build/spmm_taco`
- `benchmarks/taco/build/spadd_taco`
- `benchmarks/taco/build/spelmul_taco`
- `benchmarks/taco/build/spgemm_taco`
- `benchmarks/taco/build/sddmm_taco`

If a TACO executable is missing, the runner proceeds with only our compiler rows for that kernel.
