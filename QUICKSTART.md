# Quick Start - 3 Commands

## 1️⃣ Create DSL file

```bash
cd build

cat > example.tc << 'EOF'
tensor y : Dense<1000>;
tensor A : CSR<1000, 1000>;
tensor x : Dense<1000>;
compute y[i] = A[i, j] * x[j];
EOF
```

## 2️⃣ Compile DSL → C

```bash
./sparse_compiler example.tc -o kernel.c
```

## 3️⃣ View Generated Code

```bash
# View the optimized kernel
grep -A 10 "void spmv" kernel.c
```

```c
void spmv(const CSRMatrix* A, const double* x, double* y) {
    for (int i = 0; i < 1000; i++) {
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            int j = A->col_idx[p];
            y[i] += A->vals[p] * x[j];
        }
    }
}
```

## 4️⃣ Compile & Run Generated C Code

```bash
# Compile to executable
gcc -O2 kernel.c -o program

# Run (needs a .mtx matrix file)
./program matrix.mtx
```

---

## With Optimizations

```bash
# Loop blocking (cache tiling)
./sparse_compiler example.tc --opt-block=32 -o kernel_opt.c

# View the optimized version
grep -A 20 "void spmv" kernel_opt.c
```

---

## Complete Example

```bash
# Write DSL
echo "tensor y : Dense<100>;
tensor A : CSR<100, 100>;
tensor x : Dense<100>;
compute y[i] = A[i, j] * x[j];" > test.tc

# Compile DSL
./sparse_compiler test.tc -o test.c

# Compile C
gcc -O2 test.c -o test_program

# Done! (need a matrix.mtx to run)
```

---

## All Compiler Options

```bash
./sparse_compiler input.tc                        # Baseline (no optimizations)
./sparse_compiler input.tc --opt-block=32         # Blocking only
./sparse_compiler input.tc --opt-interchange      # Interchange only (SpMM)
./sparse_compiler input.tc --opt-all=32           # Interchange + Blocking (default order)
./sparse_compiler input.tc -o output.c            # Custom output file
./sparse_compiler --help                          # Show help

# Advanced: Optimization scheduling order (when both opts enabled)
./sparse_compiler input.tc --opt-all=32 --opt-order=I_THEN_B  # Interchange → Block (default)
./sparse_compiler input.tc --opt-all=32 --opt-order=B_THEN_I  # Block → Interchange
./sparse_compiler input.tc --opt-all=32 --opt-order=I_B_I     # Interchange → Block → Interchange
```

---

## What You Get

✅ Complete C program (226 lines)
✅ Matrix Market loader
✅ Optimized SpMV kernel
✅ Reference kernel (verification)
✅ Benchmarking (100 iterations)
✅ Timing utilities
✅ Ready to compile with gcc

---

See **TUTORIAL.md** for detailed documentation.
