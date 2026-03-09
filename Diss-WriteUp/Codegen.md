# Codegen (C/CPU Backend)

This document describes the code generator in this repository with a split between:

- **Design**: what the code generator produces and why it is structured that way.
- **Implementation**: where and how the repository implements those choices.

Scope: **targeting CPU backends via portable C** (generated `.c` files).

Primary sources:

- `include/codegen.h`, `src/codegen.cpp`
- `include/ir.h`, `src/ir.cpp` (IR shapes referenced below)
- `include/optimizations.h`, `src/optimizations.cpp` (how scheduling transforms appear in emitted code)
- `src/tests/codegen/*` (behavioral expectations)

## Design

### What We Generate

The kernel-mode generator (`codegen::CodeGenerator`) emits a **standalone C program** rather than only a kernel function. The generated C file is structured as:

1. Header comment block and `#include`s
2. Runtime struct definitions (`SparseMatrix`)
3. Matrix Market loader (COO parse + canonicalization + conversion to CSR/CSC)
4. Timing utilities
5. Matrix feature extraction (`compute_features`)
6. Optional sparse-output helpers (for kernels that output sparse matrices)
7. Optimized kernel function
8. Reference kernel function(s) (for correctness checks)
9. `main()` (argument parsing, loading, allocation, timing loop, verification, cleanup)

Why this choice:

- Keeps experiments reproducible: one artifact includes runtime + kernel + measurement harness.
- Makes kernels easy to run with `gcc/clang` without additional runtime dependencies.
- Centralizes format-correct sparse traversal in one place (codegen), while letting IR passes focus on loop structure.

In addition, a program-mode generator (`generateProgramToFile`) emits a standalone C program for a full `.tc` program containing multiple statements and `for` blocks.

### Output Model: Kernel-mode vs Program-mode

There are two materially different emission contexts:

- **Kernel-mode**: emit from one `ir::Operation`.
  - Produced by `CodeGenerator::generate` and used by `generateCode(...)` / `generateToFile(...)`.
  - Dense 2D operands are typically passed as `double**` for some kernels (notably `spmm`).
  - The loop nest is emitted by traversing the `ir::Loop` tree via the `ir::IRVisitor` interface.

- **Program-mode**: emit from `SparseTensorCompiler::IRProgram`.
  - Produced by `generateProgramToFile(...)`.
  - Dense tensors are allocated as flat `double*` buffers; 2D indexing is emitted as `A[i * A_ncols + j]`.
  - Sparse tensor element reads in AST-walk fallback are emitted via runtime helpers `sp_csr_get/sp_csc_get`.
  - If an `IRCompute` has a lowered `ir::Operation`, program-mode embeds the operation-level loop tree with a separate recursive emitter.

This split exists because:

- Program-mode needs to support arbitrary user code structure (`for` blocks, calls), not only one kernel.
- Program-mode uses flat dense buffers for simpler memory management and uniform indexing in mixed statement streams.

### Mapping IR Constructs to C

Operation-level IR is a loop tree (`ir::Loop`) plus bodies; codegen maps it into C using a small set of emission patterns.

#### Tensors and Types

- `ir::Tensor` with `Format::Dense`:
  - kernel-mode: emitted as pointers (`double*` for 1D, `double**` for some 2D kernels).
  - program-mode: emitted as `double*` always; for 2D, codegen also declares `<name>_ncols` and uses flat indexing.

- `ir::Tensor` with `Format::CSR` or `Format::CSC`:
  - emitted as `SparseMatrix*`.
  - format controls which pointer arrays are used in sparse iteration.

#### Dense Loops

`LoopKind::Dense` maps to:

```c
for (int i = lower; i < upper; i++) {
  ...
}
```

Upper bounds are often derived at runtime:

- `A->rows`, `A->cols` for sparse input dimensions
- `C_cols`, `B_cols` for dense matrix dimensions passed into the kernel
- `K` for contraction dimensions in SDDMM

#### Sparse Loops (CSR/CSC)

`LoopKind::Sparse` maps to pointer iteration over one “parent” coordinate, extracting a coordinate index from an index array.

CSR:

```c
for (int pA = A->row_ptr[parent]; pA < A->row_ptr[parent + 1]; pA++) {
  int j = A->col_idx[pA];
  ...
}
```

CSC:

```c
for (int pA = A->col_ptr[parent]; pA < A->col_ptr[parent + 1]; pA++) {
  int i = A->row_idx[pA];
  ...
}
```

The “parent” variable is format-dependent and must be chosen correctly for correctness. The generator maintains a notion of the active outer loop variable (see “Handling Sparse Iteration” below).

#### Intersection Merge (SpElMul and General IR)

If a sparse loop has `mergeStrategy == Intersection`, codegen emits a “two-pointer merge” that advances pointers in both tensors and executes the body only when the extracted coordinates match. There are CSR and CSC variants.

#### Block / Tiling Wrappers

Blocking is represented in IR by inserting a wrapper loop whose index variable name contains `"_block"` and an optional `tileBlockSize`. Codegen maps this to strip-mining:

```c
for (int i_block = 0; i_block < (UB + B - 1) / B; i_block++) {
  int i_start = i_block * B;
  int i_end = (i_start + B < UB) ? i_start + B : UB;
  for (int i = i_start; i < i_end; i++) {
    ...
  }
}
```

If a sparse loop sits directly under a block wrapper (possible under some schedules), the generator keeps the outer block loop but enforces the bounded dense range via a bounds override, rather than rewriting sparse pointer bounds.

#### Loop Bodies

Loop bodies exist in two forms:

- **Structured** (`IRStmt`/`IRExpr`) for some lowering paths:
  - codegen emits final C syntax directly from the nodes.
  - used when `Loop::hasStructuredBody()` is true.

- **String bodies** (legacy, common in kernel-specific builders):
  - the loop body is a C-like snippet stored in `Loop::body` (and setup code in `Loop::preBody`).
  - codegen does placeholder rewriting for sparse values: `{tensor}_vals[` becomes `{tensor}->vals[`.

### Memory Layout Translation

#### SparseMatrix Runtime Layout

Generated C defines:

```c
typedef struct {
  int rows, cols, nnz;
  int* row_ptr; int* col_idx;  // CSR
  int* col_ptr; int* row_idx;  // CSC (same memory, different interpretation)
  double* vals;
} SparseMatrix;
```

The loader and converters populate these fields such that:

- CSR interpretation uses `row_ptr/col_idx/vals`.
- CSC interpretation uses `col_ptr/row_idx/vals`.

Many generic helpers (feature extraction, some loops) use `row_ptr/col_idx` and rely on the convention that for CSC, `row_ptr == col_ptr` and `col_idx == row_idx`.

#### MatrixMarket Loader: COO → CSR/CSC

The loader:

- Parses Matrix Market coordinate files, supports `real`, `integer`, and `pattern`.
- Expands symmetric / skew-symmetric entries as needed.
- Sorts entries in a canonical order:
  - CSR target: `(row, col)`
  - CSC target: `(col, row)`
- Merges duplicate coordinates by summing values.
- Converts to the target sparse format (emits either CSR or CSC conversion code).

This makes baselines deterministic and keeps sparse iteration stable across runs.

#### Dense Layout Choices

Two dense representations show up depending on emission mode:

- kernel-mode: some kernels (e.g. SpMM) use `double**` for 2D dense operands and output to keep the code close to textbook formulations.
- program-mode: dense 2D tensors are always flat `double*`; indexing is rewritten to `A[i * A_ncols + j]`.

### Handling Sparse Iteration Correctly

Sparse loops require a correct “parent” coordinate for pointer bounds (`row_ptr[parent]` in CSR, `col_ptr[parent]` in CSC). Codegen manages this with:

- `currentOuterLoopVar_`: tracks which outer coordinate is currently binding the sparse dimension.
- `parentVarOverride`: optional per-loop override when a nested sparse loop must use a specific parent variable (used by certain SpGEMM variants).
- `loopBoundsOverride_`: a map of dense index names to `(start,end)` strings to enforce bounded iteration ranges under block wrappers and certain schedules.

This design keeps sparse pointer bounds format-correct even under interchange and blocking transformations.

### Code Emission Strategy and Traversal

#### Kernel-mode traversal (visitor)

`CodeGenerator` implements `ir::IRVisitor` and emits a loop tree by calling `rootLoop->accept(*this)`. `visitLoop(...)` dispatches:

- `*_block` loop wrapper: `emitBlockLoop`
- dense loop: `emitDenseLoop`
- sparse loop with intersection merge: `emitMergeIntersectionCSR/CSC`
- sparse loop without merge: `emitSparseLoopCSR/emitSparseLoopCSC` (format chosen from tensor metadata)

#### Program-mode traversal (recursive emitter)

Program-mode has a separate recursive emitter (e.g. `emitLoopTree`) that:

- emits `ir::Loop` trees inside a program statement stream,
- respects `Loop::isExternallyBound` so loops already bound by an enclosing program-level `for` are not re-emitted,
- provides program-mode runtime bounds and dense indexing rules.

### Variable Management and Naming Conventions

Generated code uses predictable naming, which is relied upon by both string-body templates and codegen transformations:

- sparse pointers: `p<tensorName>` (e.g. `pA`, `pB`, `pS`, `pC`)
- block wrappers:
  - wrapper var: `<idx>_block`
  - bounds vars: `<idx>_start`, `<idx>_end`
- extracted coordinates:
  - the loop’s index variable name is emitted directly (e.g. `int j = A->col_idx[pA];`)

Kernel-specific temporaries also use stable names:

- SpMM blocked variants: `acc[...]`, `btile[...]`, `jj`, `w`, `t`
- SpGEMM hash-per-row: `acc`, `marked`, `touched`, `touched_count`, plus hash structures

### Kernel Generation and CPU Backend Targeting

The generated code targets CPU execution:

- Uses standard C library facilities (`malloc/calloc/free`, `memcpy/memset`, `qsort`, `<math.h>`).
- Uses `clock_gettime(CLOCK_MONOTONIC)` for timing.
- No threading, SIMD-specific intrinsics, or GPU runtime in current codegen.

Optimizations are expressed as loop-structure changes and kernel-local temporaries rather than backend-specific instructions.

## Implementation

### High-level API and Entry Points

Declared in `include/codegen.h`, implemented in `src/codegen.cpp`:

- `codegen::generateCode(const ir::Operation&, const opt::OptConfig&) -> std::string`
- `codegen::generateToFile(const ir::Operation&, const opt::OptConfig&, filename) -> bool`
- `codegen::generateProgramToFile(const SparseTensorCompiler::IRProgram&, ...) -> bool`
- `codegen::CodeGenerator::generate(...)`: emits the full C program for a single operation.

### Kernel-mode emission pipeline (`CodeGenerator::generate`)

`CodeGenerator::generate(...)` emits sections in order:

1. `emitHeader()`
2. `emitStructDefinitions()` (`SparseMatrix`)
3. `emitMatrixMarketLoader()` (COO parser + calls `emitCSRConversion`/`emitCSCConversion`)
4. `emitTimingHarness()`
5. `emitFeatureExtraction()` (`compute_features`)
6. Sparse-output helpers if needed:
   - `emitSparseOutputHelpers()` when `isSparseOutputMode()`
   - legacy hash helpers for hash-per-row path (mostly for SpGEMM)
7. `emitKernel()` (optimized kernel function)
8. `emitReferenceKernel()` (naive reference implementation; used by verification in `main`)
9. `emitMain()` (argument parsing, allocation, timing loop, reporting, verification, cleanup)

### Kernel function emission (`emitKernel`)

`emitKernel()`:

- optionally emits sparse assembly functions first via `emitSparseAssemblyForKernel()` when output is sparse.
- emits `void <kernelSignature> { ... }`, where the signature comes from `getKernelSignature()`.
- prints comments indicating which optimizations were applied (reordering, interchange, blocking).
- emits any `currentOp_->prologueLines` (workspace alloc) and `epilogueLines` (workspace free).
- traverses the loop tree via the IR visitor:
  - `currentOp_->rootLoop->accept(*this)`

Note: there are legacy/specialized SpMM blocked emitters (`emitSpMMBlockedCSR/emitSpMMBlockedCSC`). The current unified pipeline emits SpMM via the loop-tree visitor path; these functions remain as alternative emission patterns.

### Loop emission details (kernel-mode)

Key loop emitters in `src/codegen.cpp`:

- `emitDenseLoop(const ir::Loop&)`
  - chooses runtime upper bounds based on `currentOp_->kernelType` and the loop variable name.
  - uses `loopBoundsOverride_` when a block wrapper needs to constrain iteration.
  - maintains `currentOuterLoopVar_` to keep sparse loops correctly parented.

- `emitSparseLoopCSR(const ir::Loop&)` and `emitSparseLoopCSC(const ir::Loop&)`
  - decide `tensorName` from `loop.sparseTensorName`
  - select parent variable from `parentVarOverride` or `currentOuterLoopVar_`
  - emit pointer loop and extracted coordinate assignment
  - emit `preBody/preStmts`, then children, then `body/postStmts`

- `emitMergeIntersectionCSR/emitMergeIntersectionCSC(const ir::Loop&)`
  - used when `loop.mergeStrategy == Intersection`
  - emits two-pointer merge logic over two sparse inputs

- `emitBlockLoop(const ir::Loop&)`
  - recognizes wrapper loops by `"_block"` suffix in the index name
  - supports per-wrapper block sizes via `loop.tileBlockSize`
  - binds bounded iteration ranges and interacts with sparse children via `loopBoundsOverride_`

- `emitLoopBody(const ir::Loop&)`
  - structured path emits `IRStmt` nodes via `emitIRStmt`
  - string path rewrites `{tensor}_vals[` to `tensor->vals[` and emits the result

### Structured statement/expression emission

Declared in `include/codegen.h` and implemented in `src/codegen.cpp`:

- `IRExprEmitter` (visitor) emits final, codegen-ready C expressions from `ir::IRExpr`.
- `CodeGenerator::emitIRExpr` and `CodeGenerator::emitIRStmt` emit `IRStmt` nodes (`IRScalarDecl`, `IRAssign`, `IRCallStmt`).

This path avoids placeholder rewriting and is the intended end-state for general kernels and more structured lowering.

### Sparse-output support

When `isSparseOutputMode()` is true, codegen emits:

- assembly helpers (`emitSparseAssemblyForKernel` dispatches to `emitSpAddSparseAssemble`, `emitSpElMulSparseAssemble`, `emitSpGEMMSparseAssemble`, `emitSDDMMSparseAssemble`)
- runtime helpers (`emitSparseOutputHelpers`), such as:
  - `zero_sparse_values(C)` to reset `C->vals`
  - sparse error metrics helpers

### Reference kernels and verification

`emitReferenceKernel()` emits naive/reference implementations that follow the sparse format (CSR vs CSC) for correctness checking. `emitMain()` uses these for error measurement (for kernels that include verification in main).

### Program-mode codegen (`generateProgramToFile`)

Program-mode codegen is implemented as a separate path in `src/codegen.cpp` (anonymous-namespace helpers):

- `emitProgExpr(...)` emits expressions from AST nodes:
  - sparse 2D access uses `sp_csr_get(A,row,col)` or `sp_csc_get(A,row,col)`
  - dense 2D access uses flat indexing with `<name>_ncols`
- `emitForLoopC(...)` emits `IRForLoop` headers and body recursively.
- When an `IRCompute` has a lowered `ir::Operation`:
  - `emitKernelFromOperation` and `emitLoopTree` emit the operation-level loop tree in program-mode, honoring `Loop::isExternallyBound`.

This path is the bridge between “DSL programs” and “operation-level optimized kernels”.

### Tests as spec

The following tests in `src/tests/codegen/` encode expectations about generated output and are a good reference for what must stay stable:

- kernel identification and signatures: `test_spmv_codegen.cpp`, `test_spmm_codegen.cpp`, `test_sddmm_codegen.cpp`, `test_spadd_codegen.cpp`, `test_spelmul_codegen.cpp`, `test_spgemm_codegen.cpp`
- loop transformation emission and correctness: `test_blocking_codegen.cpp`, `test_scheduled_optimizations_correctness.cpp`, `test_reordering_correctness.cpp`
- sparse-output emission: `test_sparse_output_codegen.cpp`, `test_spgemm_hash_output.cpp`
- unified visitor path coverage: `test_visitor_unification.cpp`

