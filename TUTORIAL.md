# SparseTensorCompiler - Quick Start Tutorial

## End-to-End: From DSL to Executable

This tutorial shows you how to use the SparseTensorCompiler to generate optimized C code from tensor algebra DSL.

---

## Prerequisites

```bash
cd /Users/ilmihakam/co/backend/SparseTensorCompiler/build
```

Make sure the compiler is built:
```bash
ls sparse_compiler  # Should exist
```

Check the compiler version:
```bash
./sparse_compiler --version
# Output: SparseTensorCompiler v0.1.0
```

---

## Step 1: Write Your DSL Input

Create a file `my_spmv.tc` with your tensor computation:

```bash
cat > my_spmv.tc << 'EOF'
tensor y : Dense<1000>;
tensor A : CSR<1000, 1000>;
tensor x : Dense<1000>;
compute y[i] = A[i, j] * x[j];
EOF
```

**DSL Syntax:**
- `tensor name : Format<dims>` - Declare tensors
- `compute output[indices] = expression` - Define computation
- Formats: `Dense`, `CSR` (Compressed Sparse Row), `CSC` (Compressed Sparse Column)
- Einstein notation: repeated indices are summed

---

## Step 2: Compile DSL → C Code

### Basic Usage (No Optimizations)

```bash
./sparse_compiler my_spmv.tc -o my_kernel.c
```

**Output:**
```
✓ Parse successful
✓ IR generated
  Interchange: OFF
  Blocking: OFF
✓ Optimizations applied
✓ C code generated

============================================================
✓ COMPILATION SUCCESSFUL
============================================================

Output: my_kernel.c (6148 bytes)

Next steps:
  1. Compile to executable:
     gcc -O2 my_kernel.c -o program

  2. Run with a matrix:
     ./program <matrix.mtx>

Tip: Try --opt-all=32 for optimized performance
```

### With Optimizations

**Blocking (cache tiling):**
```bash
./sparse_compiler my_spmv.tc --opt-block=32 -o my_kernel_blocked.c
```

**All optimizations:**
```bash
./sparse_compiler my_spmv.tc --opt-all=32 -o my_kernel_optimized.c
```

**Verbose mode (detailed output):**
```bash
./sparse_compiler my_spmv.tc --opt-all=32 -v -o my_kernel.c
```

Output with verbose mode:
```
→ Reading input file: my_spmv.tc
  File size: 107 bytes
→ Parsing DSL
✓ Parse successful
→ Lowering AST to IR
  Kernel type: spmv
  Input tensors: 2
✓ IR generated
→ Applying optimizations
  Interchange: ON
  Blocking: ON (size=32)
  Order: I_THEN_B (Interchange → Block)
  ✓ Blocking applied (size=32)
✓ Optimizations applied
→ Generating C code
  Output size: 6533 bytes
✓ C code generated

============================================================
✓ COMPILATION SUCCESSFUL
============================================================
```

**Available options:**
```bash
./sparse_compiler --help
```

**Quick reference:**
- `-o <file>` - Specify output file
- `--opt-interchange` - Enable loop interchange (SpMM)
- `--opt-block=SIZE` - Enable blocking with block size
- `--opt-all=SIZE` - Enable all optimizations
- `--opt-order=ORDER` - Set optimization order (I_THEN_B, B_THEN_I, I_B_I)
- `-v, --verbose` - Show detailed compilation steps
- `--version` - Show version information
- `-h, --help` - Show help message

---

## Step 3: View the Generated C Code

```bash
# View the entire file
cat my_kernel.c

# View just the kernel (first 100 lines)
head -100 my_kernel.c

# View the optimized kernel section
grep -A 20 "void spmv" my_kernel.c
```

**What's in the generated C file:**
- Complete standalone C program (~226 lines)
- CSRMatrix structure definition
- Matrix Market (.mtx) file loader
- Optimized SpMV kernel (your computation)
- Reference kernel (for verification)
- Timing utilities
- Main function with benchmarking

---

## Step 4: Compile the Generated C Code

```bash
gcc -O2 my_kernel.c -o my_spmv_program
```

This produces an executable `my_spmv_program` (~34KB).

---

## Step 5: Run Your Program

### Create a Test Matrix (Matrix Market format)

```bash
cat > test_matrix.mtx << 'EOF'
%%MatrixMarket matrix coordinate real general
1000 1000 2000
1 1 1.0
2 2 2.0
3 3 3.0
EOF

# Add more entries (diagonal + off-diagonal)
for i in {4..1000}; do echo "$i $i $i.0" >> test_matrix.mtx; done
for i in {1..999}; do echo "$i $((i+1)) 0.5" >> test_matrix.mtx; done
```

### Run the Benchmark

```bash
./my_spmv_program test_matrix.mtx
```

**Expected Output:**
```
Matrix: test_matrix.mtx (1000 x 1000, 2000 nnz)
Iterations: 100
Total time: 0.15 ms
Avg time per iteration: 0.0015 ms
Max error vs reference: 0.000000e+00
```

**Output Explanation:**
- Shows matrix dimensions and number of non-zeros
- Runs 100 iterations for accurate timing
- Verifies correctness against reference implementation
- `Max error = 0` means the optimized kernel is correct!

---

## Complete Example Workflow

```bash
# 1. Write DSL
echo "tensor y : Dense<1000>;
tensor A : CSR<1000, 1000>;
tensor x : Dense<1000>;
compute y[i] = A[i, j] * x[j];" > spmv.tc

# 2. Compile to C (baseline)
./sparse_compiler spmv.tc -o spmv_baseline.c

# 3. Compile to C (optimized)
./sparse_compiler spmv.tc --opt-block=32 -o spmv_optimized.c

# 4. View the kernels
echo "=== BASELINE KERNEL ==="
grep -A 10 "void spmv" spmv_baseline.c | head -15

echo -e "\n=== OPTIMIZED KERNEL ==="
grep -A 15 "void spmv" spmv_optimized.c | head -20

# 5. Compile both versions
gcc -O2 spmv_baseline.c -o spmv_baseline
gcc -O2 spmv_optimized.c -o spmv_optimized

# 6. Create test matrix
cat > matrix.mtx << 'EOF'
%%MatrixMarket matrix coordinate real general
1000 1000 3000
EOF
for i in {1..1000}; do echo "$i $i 1.0" >> matrix.mtx; done
for i in {1..999}; do echo "$i $((i+1)) 0.5" >> matrix.mtx; done
for i in {2..1000}; do echo "$i $((i-1)) 0.5" >> matrix.mtx; done

# 7. Run and compare
echo "=== Baseline ==="
./spmv_baseline matrix.mtx

echo -e "\n=== Optimized ==="
./spmv_optimized matrix.mtx
```

---

## Optimization Modes Explained

### 1. Baseline (No Optimizations)
```bash
./sparse_compiler input.tc -o output.c
```
- Generates straightforward nested loops
- Good for understanding the code structure
- Baseline for performance comparison

### 2. Loop Blocking (Cache Tiling)
```bash
./sparse_compiler input.tc --opt-block=32 -o output.c
```
- Tiles the outer loop into blocks
- Improves cache locality
- Block size 32 fits in L1 cache (~1KB)
- Try 16, 32, 64 for different matrices

### 3. Loop Interchange
```bash
./sparse_compiler input.tc --opt-interchange -o output.c
```
- Reorders loop nesting for cache locality (SpMM)
- Improves memory access patterns in dense B/C accesses

### 4. All Optimizations
```bash
./sparse_compiler input.tc --opt-all=32 -o output.c
```
- Combines interchange + blocking
- Best performance for most matrices

### 5. Optimization Scheduling Order (Advanced)
```bash
./sparse_compiler input.tc --opt-all=32 --opt-order=I_THEN_B -o output.c
./sparse_compiler input.tc --opt-all=32 --opt-order=B_THEN_I -o output.c
./sparse_compiler input.tc --opt-all=32 --opt-order=I_B_I -o output.c
```

When both optimizations are enabled, you can control their application order:

- **I_THEN_B** (default): Interchange → Block
  - Apply interchange first, then cache tiling
  - Best for most cases

- **B_THEN_I**: Block → Interchange
  - Apply cache tiling first, then interchange
  - Experimental, may produce different performance

- **I_B_I**: Interchange → Block → Interchange
  - Apply interchange twice with blocking in between
  - For complex transformations

**Usage:**
```bash
# Default order (same as I_THEN_B)
./sparse_compiler spmv.tc --opt-all=32 -o output.c

# Explicit order specification
./sparse_compiler spmv.tc --opt-all=32 --opt-order=B_THEN_I -o output.c

# Generate all three for benchmarking
./sparse_compiler spmv.tc --opt-all=32 --opt-order=I_THEN_B -o i_then_b.c
./sparse_compiler spmv.tc --opt-all=32 --opt-order=B_THEN_I -o b_then_i.c
./sparse_compiler spmv.tc --opt-all=32 --opt-order=I_B_I -o i_b_i.c
```

The generated C code will show which order was used:
```c
// Optimization order: B_THEN_I (Block → Interchange)
```

**Note:** Order only matters when both interchange and blocking are enabled. With only one optimization, the order flag has no effect.

---

## Viewing Generated Code Differences

Compare baseline vs optimized:

```bash
# Generate both versions
./sparse_compiler input.tc -o baseline.c
./sparse_compiler input.tc --opt-block=32 -o optimized.c

# Compare the kernels
diff <(grep -A 20 "void spmv" baseline.c) \
     <(grep -A 20 "void spmv" optimized.c)
```

Or use a visual diff tool:
```bash
code --diff baseline.c optimized.c  # VS Code
```

---

## File Structure

```
SparseTensorCompiler/
├── build/
│   ├── sparse_compiler          # The compiler executable
│   ├── my_spmv.tc          # Your DSL input
│   ├── my_kernel.c         # Generated C code
│   └── my_spmv_program     # Compiled executable
├── examples/
│   └── spmv_simple.tc      # Example DSL files
└── TUTORIAL.md             # This file
```

---

## Troubleshooting

**Parse errors:**
- Remove comments from .tc files (not supported yet)
- Check syntax: `tensor name : Format<size>;`
- Ensure compute statement uses valid indices

**Compilation errors:**
- Make sure gcc is installed: `gcc --version`
- Check for C99/C11 compatibility

**Runtime crashes:**
- Matrix dimensions must match DSL declaration
- Currently, DSL dimensions are compile-time (limitation)

---

## What Gets Generated

Every generated C file includes:

1. **Header Section** (~10 lines)
   - Shows which optimizations are enabled
   - Standard C includes

2. **CSRMatrix Structure** (~10 lines)
   - row_ptr, col_idx, vals arrays

3. **Matrix Market Loader** (~120 lines)
   - Reads .mtx files
   - Converts COO → CSR format

4. **Timing Utilities** (~10 lines)
   - High-resolution timing (clock_gettime)

5. **Optimized Kernel** (~10-30 lines)
   - Your computation with optimizations

6. **Reference Kernel** (~15 lines)
   - Naive implementation for verification

7. **Main Function** (~60 lines)
   - Argument parsing
   - Matrix loading
   - Warmup + benchmarking (100 iterations)
   - Correctness verification
   - Results reporting

**Total: ~226 lines of production-ready C code**

---

## Benchmarking Workflow

This section shows how to systematically benchmark all optimization configurations to find which works best for your matrices.

### Step 1: Generate All Configurations

Create all optimization variants from a single DSL file:

```bash
# Baseline (no optimizations)
./sparse_compiler spmv.tc -o baseline.c

# Individual optimizations
./sparse_compiler spmv.tc --opt-interchange -o interchange_only.c
./sparse_compiler spmv.tc --opt-block=32 -o block_only.c

# All optimizations with different orders
./sparse_compiler spmv.tc --opt-all=32 --opt-order=I_THEN_B -o i_then_b.c
./sparse_compiler spmv.tc --opt-all=32 --opt-order=B_THEN_I -o b_then_i.c
./sparse_compiler spmv.tc --opt-all=32 --opt-order=I_B_I -o i_b_i.c
```

Or use a loop for efficiency:
```bash
# Generate all scheduling orders
for order in I_THEN_B B_THEN_I I_B_I; do
    ./sparse_compiler spmv.tc --opt-all=32 --opt-order=$order -o spmv_${order}.c
done
```

### Step 2: Compile All Versions

```bash
gcc -O2 baseline.c -o baseline
gcc -O2 interchange_only.c -o interchange_only
gcc -O2 block_only.c -o block_only
gcc -O2 i_then_b.c -o i_then_b
gcc -O2 b_then_i.c -o b_then_i
gcc -O2 i_b_i.c -o i_b_i
```

### Step 3: Create Test Matrices

Download from SuiteSparse or create test matrices:

```bash
# Create a simple test matrix (1000x1000, tridiagonal)
cat > test.mtx << 'EOF'
%%MatrixMarket matrix coordinate real general
1000 1000 2998
EOF

# Diagonal
for i in {1..1000}; do echo "$i $i 2.0" >> test.mtx; done

# Off-diagonals
for i in {1..999}; do
    echo "$i $((i+1)) -1.0" >> test.mtx
    echo "$((i+1)) $i -1.0" >> test.mtx
done
```

### Step 4: Run Benchmarks

Run all versions on the same matrix:

```bash
echo "Configuration,Time(ms),Error"
./baseline test.mtx | grep "Avg time" | awk '{print "Baseline," $5}'
./interchange_only test.mtx | grep "Avg time" | awk '{print "Interchange," $5}'
./block_only test.mtx | grep "Avg time" | awk '{print "Block," $5}'
./i_then_b test.mtx | grep "Avg time" | awk '{print "I_THEN_B," $5}'
./b_then_i test.mtx | grep "Avg time" | awk '{print "B_THEN_I," $5}'
./i_b_i test.mtx | grep "Avg time" | awk '{print "I_B_I," $5}'
```

### Step 5: Automated Benchmarking Script

Create a comprehensive benchmarking script:

```bash
#!/bin/bash
# benchmark_all.sh

MATRICES="test.mtx matrix1.mtx matrix2.mtx"
CONFIGS="baseline interchange_only block_only i_then_b b_then_i i_b_i"

echo "Matrix,Config,Time(ms),Speedup"

for matrix in $MATRICES; do
    # Get baseline time
    baseline_time=$(./baseline $matrix 2>&1 | grep "Avg time" | awk '{print $5}')

    for config in $CONFIGS; do
        time=$(./$config $matrix 2>&1 | grep "Avg time" | awk '{print $5}')
        speedup=$(echo "scale=2; $baseline_time / $time" | bc)
        echo "$matrix,$config,$time,${speedup}x"
    done
done
```

Run the script:
```bash
chmod +x benchmark_all.sh
./benchmark_all.sh > results.csv
```

### Step 6: Analyze Results

```bash
# View results
column -t -s',' results.csv

# Find best configuration per matrix
sort -t',' -k1,1 -k4,4nr results.csv | awk -F',' '!seen[$1]++'
```

### Example Output

```
Matrix       Config      Time(ms)  Speedup
test.mtx     baseline    0.0374    1.00x
test.mtx     reorder     0.0365    1.02x
test.mtx     block       0.0298    1.26x
test.mtx     I_THEN_B    0.0285    1.31x
test.mtx     B_THEN_I    0.0292    1.28x
test.mtx     I_B_I       0.0288    1.30x
```

### Key Insights from Benchmarking

1. **Different orders have different performance** - Some matrices benefit more from one order than another
2. **Block size matters** - Try 16, 32, 64 for different cache sizes
3. **Matrix characteristics** - Dense regions benefit from blocking, sparse regions from reordering
4. **Systematic testing** - Use this workflow for Milestone 5 research

---

## Next Steps

1. **Try different matrices:**
   - Download from SuiteSparse: https://sparse.tamu.edu/
   - Convert to Matrix Market format if needed

2. **Experiment with optimizations:**
   - Try different block sizes: 16, 32, 64, 128
   - Compare performance with/without optimizations

3. **Benchmark real matrices:**
   - Large sparse matrices show bigger performance differences
   - Dense matrices don't benefit from blocking as much

4. **Extend the DSL:**
   - Try SpMM: `compute C[i, j] = A[i, k] * B[k, j];`
   - Experiment with CSC format

---

## Summary

### Basic Workflow (Three Commands)
```bash
# 1. Compile DSL to C
./sparse_compiler my_computation.tc -o kernel.c

# 2. Compile C to executable
gcc -O2 kernel.c -o program

# 3. Run
./program matrix.mtx
```

### Advanced Features Available
- ✅ **Multiple optimization modes**: baseline, reordering, blocking, or both
- ✅ **Optimization scheduling**: Control order (I_THEN_B, B_THEN_I, I_B_I)
- ✅ **Verbose mode**: See detailed compilation steps with `-v`
- ✅ **Professional CLI**: Colored output, progress indicators, helpful tips
- ✅ **Benchmarking-ready**: Easy to generate and compare all configurations
- ✅ **Complete help**: Run `./sparse_compiler --help` for all options

### For Benchmarking Research
Use the automated workflow in the "Benchmarking Workflow" section to:
1. Generate all optimization configurations automatically
2. Compile all versions
3. Run systematic benchmarks
4. Collect and analyze performance data
5. Determine which optimization order works best for your matrices

**That's it!** You now have a production-ready sparse tensor compiler with systematic benchmarking capabilities! 🎉

See **docs/MILESTONE4_COMPLETE.md** for complete CLI documentation.
