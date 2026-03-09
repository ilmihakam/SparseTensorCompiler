# Codegen (C/CPU Backend)

This document describes the current code generation architecture for the CPU backend.
It is split into:

- **Design**: the intended boundary between kernel emission and surrounding generated C.
- **Implementation**: how that boundary is represented in the repository.
- **Harness Functionality**: what the non-kernel emitters provide around the generated compute kernel.

Scope: **portable C code generation for CPU execution**.

Primary sources:

- `include/codegen.h`
- `include/semantic_ir.h`
- `src/codegen.cpp`
- `src/codegen/runtime_emitter.cpp`
- `src/codegen/reference_emitter.cpp`
- `src/codegen/output_assembly_emitter.cpp`
- `src/codegen/output_assembly_shared.cpp`
- `src/codegen/program_emitter.cpp`
- `src/tests/codegen/*`, `src/tests/ir/*`, `src/tests/optimizations/*`

## Design

### Core Architectural Boundary

The code generation architecture is now split by responsibility.

The central rule is:

- if a piece of emitted C is **derivable from scheduled compute IR**, it belongs in `codegen.cpp`
- if it is **not derivable from scheduled compute IR**, it belongs in a separate emitter

Concretely, `codegen.cpp` is now responsible only for **IR-derivable compute emission**:

- the compute kernel signature
- compute prologue and epilogue statements
- traversal of the scheduled loop tree
- emission of loop-local bindings
- emission of structured `IRStmt` / `IRExpr` inside the compute body

This means the core backend emits a fragment of the form:

```c
void compute(...) {
    ...
}
```

rather than owning the whole standalone C file.

### Compute Emission Contract

The compute emitter consumes:

- `scheduled::Compute`
- `scheduled::Loop`
- attached `IRStmt`
- attached `IRExpr`

The loop contract is explicit. `scheduled::Loop` no longer exposes legacy codegen-policy fields such as driver tensor names or merge tensor lists as part of the active emission interface. Instead, emission is guided by:

- `LoopHeaderKind`
- `lowerExpr` / `upperExpr`
- `bindingVarName` / `bindingExpr`
- `SparseIteratorEmission`
- `MergeEmission`
- `BlockEmission`

The current loop header forms are:

- `DenseFor`
- `SparseIterator`
- `SparseMerge`
- `Block`

This moves the responsibility split to the correct place:

- lowering/scheduling decides traversal semantics
- `scheduled::Loop` stores those semantics explicitly
- `codegen.cpp` renders them as C

### Loop Forms in Generated C

Dense loops are emitted from explicit bound expressions:

```c
for (int j = lower; j < upper; j++) {
    ...
}
```

Sparse iterator loops are emitted from explicit pointer-begin, pointer-end, and binding expressions:

```c
for (int pA = A->col_ptr[k]; pA < A->col_ptr[k + 1]; pA++) {
    int i = A->row_idx[pA];
    ...
}
```

Sparse merge loops use an explicit merge descriptor:

- union and intersection are represented in scheduled IR via `merge.strategy`
- each merged tensor contributes explicit pointer and candidate-index expressions in `merge.terms`

Block loops use explicit strip-mining metadata:

```c
for (int j_block = 0; j_block < (N_j + 31) / 32; j_block++) {
    int j_start = j_block * 32;
    int j_end = (j_start + 32 < N_j) ? j_start + 32 : N_j;
    ...
}
```

The bounded inner dense loop uses the block descriptor rather than a codegen-side override map.

### Full-Program Generation

Standalone `.c` generation still exists, but it is no longer conceptually owned by `codegen.cpp`.

Instead, a full generated C file is composed from multiple emitters:

- compute kernel emitter
- runtime/harness emitter
- reference emitter
- sparse-output assembly emitter
- scheduled-program emitter

This keeps the kernel path structurally tied to scheduled IR while still supporting benchmarkable standalone artifacts.

### What Is Still Transitional

The main architecture is now in place:

- compute emission is driven by explicit scheduled-loop metadata
- non-kernel responsibilities are separated into dedicated emitters

Some auxiliary abstractions are still transitional:

- access-expression metadata in `IRTensorAccess` is still somewhat codegen-oriented
- some runtime/signature metadata is still derived in `EmissionContext`
- program and runtime emitters remain more string-oriented than the compute emitter

These do not invalidate the core architecture, but they are remaining cleanup areas rather than unresolved architectural questions.

## Implementation

### Public Entry Points

The public API is declared in `include/codegen.h`.

The main entry points are:

- `generateCode(...)`
- `generateKernelCode(...)`
- `generateToFile(...)`
- `generateKernelToFile(...)`
- `generateProgramToFile(...)`

The intended use is:

- `generateKernelCode(...)`: emit only the compute kernel function
- `generateCode(...)`: emit a full standalone C program for one scheduled compute
- `generateProgramToFile(...)`: emit a standalone C program for a scheduled multi-statement program

### `src/codegen.cpp`: Compute Kernel Emitter

`src/codegen.cpp` now holds the compute-emission core.

Key responsibilities:

- compute emission setup/reset for `scheduled::Compute`
- `generateKernel(...)`
- `emitInlineScheduledCompute(...)`
- recursive loop-tree emission from `scheduled::Loop`
- `emitIRStmt(...)` / `emitIRExpr(...)`
- structural compute-signature generation

The compute loop emitter reads explicit loop metadata:

- `headerKind`
- `lowerExpr`
- `upperExpr`
- `iterator`
- `merge`
- `block`

The old codegen-side loop-policy state has been removed from the compute path. In particular, loop rendering no longer relies on mutable outer-variable tracking or dense-bound override maps stored on the generator object.

### `src/codegen/runtime_emitter.cpp`: Full-Program Wrapper for Single Computes

`runtime_emitter.cpp` now owns the full-program path for a single `scheduled::Compute`.

Responsibilities:

- `CodeGenerator::generate(...)`
- `generateCode(...)`
- `generateToFile(...)`
- header emission
- `SparseMatrix` definition
- Matrix Market loader and conversions
- timing/statistics helpers
- feature extraction
- `main()`

This module orchestrates standalone C generation around the kernel emitter, but it does not define compute-loop structure itself.

### `src/codegen/reference_emitter.cpp`: Reference Kernels

`reference_emitter.cpp` contains the non-optimized/reference-kernel path.

Responsibilities:

- `emitReferenceKernel()`
- scheduled reference traversal for dense-output cases
- sparse-output-aware reference implementations for:
  - union
  - intersection
  - sampled output
  - dynamic-row accumulation

This keeps correctness-checking logic separate from optimized kernel emission.

### `src/codegen/output_assembly_emitter.cpp` and `output_assembly_shared.cpp`

Sparse-output support is split into:

- `output_assembly_emitter.cpp`
  - standalone wrapper emission for sparse-output helpers
- `output_assembly_shared.cpp`
  - shared assembly-body logic reused by both standalone and program-mode paths

This subsystem handles sparse-output assembly concerns such as:

- output structure assembly
- zeroing/reset helpers
- sparse helper routines used by standalone generated programs

It is intentionally outside the compute-kernel emitter boundary.

### `src/codegen/program_emitter.cpp`: Scheduled Program Emission

`program_emitter.cpp` handles full scheduled programs rather than one compute kernel.

Responsibilities:

- `generateProgramToFile(...)`
- emission of regions, calls, and mixed program statements
- embedding scheduled compute regions through `emitInlineScheduledCompute(...)`
- program-mode expression emission and top-level orchestration

This path supports full `.tc` programs with multiple statements and user-level control structure, rather than only one kernel-shaped computation.

### Scheduled Loop Contract in the Repository

The explicit emission contract is represented in `include/semantic_ir.h`.

Important types:

- `LoopHeaderKind`
- `SparseIteratorEmission`
- `MergeTermEmission`
- `MergeEmission`
- `BlockEmission`

Lowering populates these descriptors in `src/semantic_ir.cpp`.
Optimization passes that rewrite loop trees preserve them in `src/scheduled_optimizations.cpp`.

That gives the compute emitter a clear contract:

- lowering computes loop semantics once
- scheduling/optimization preserves those semantics through loop-tree rewrites
- codegen renders them without re-deriving format-specific mechanics

### Tests as Specification

The current architecture is backed by tests at multiple levels:

- kernel text and parity:
  - `src/tests/codegen/test_kernel_golden.cpp`
  - `src/tests/codegen/test_codegen_framework.cpp`
- scheduled-loop contract:
  - `src/tests/ir/test_semantic_ir.cpp`
  - `src/tests/ir/test_general_lowering.cpp`
  - `src/tests/ir/test_new_kernel_lowering.cpp`
  - `src/tests/ir/test_spmm_ir_lowering.cpp`
- optimization preservation of loop metadata:
  - `src/tests/optimizations/test_blocking_pass.cpp`
  - `src/tests/optimizations/test_new_kernel_optimizations.cpp`
  - `src/tests/optimizations/test_sparse_output_config_effects.cpp`

These tests verify both emitted C and the scheduled-loop metadata consumed by the kernel emitter.

## Harness Functionality

### Runtime Support

The non-kernel emitters provide the runtime environment needed to compile and execute standalone generated C files.

This includes:

- the `SparseMatrix` runtime struct
- Matrix Market parsing
- COO canonicalization and CSR/CSC conversion
- dense and sparse allocation helpers used by the generated program
- timing and statistics reporting
- matrix feature extraction

These are experiment-facing concerns, not IR-derived compute concerns.

### Reference and Verification

Standalone generated programs can include reference kernels and verification support.

This functionality exists so generated kernels can be:

- timed
- compared against a reference implementation
- checked for correctness in the same generated artifact

That logic is intentionally split from optimized compute emission so that the compute emitter remains tied only to scheduled IR.

### Sparse-Output Support

Sparse-output kernels need additional helper functionality beyond the compute kernel body.

Examples:

- sparse pattern assembly
- value reset helpers
- dynamic-row or hash-based accumulation support
- sparse-output verification helpers

These helpers are not derivable from the compute loop tree alone, so they live in the output-assembly subsystem rather than in `codegen.cpp`.

### Program-Level Orchestration

For full DSL programs, the harness layer also includes:

- statement-level orchestration
- scheduled regions
- emitted calls
- allocation and cleanup around multiple compute regions

This is why `program_emitter.cpp` exists as a separate layer: it composes compute emission into a larger generated program without turning the compute emitter into a second program frontend.

### Practical Role of the Harness Layer

The harness emitters serve three main purposes:

- make generated kernels runnable as standalone C artifacts
- support benchmarking and correctness evaluation
- keep non-IR-derived concerns out of the compute-kernel emitter

This split is the key architectural outcome of the refactor:

- `codegen.cpp` handles IR-derived compute emission
- the harness layers handle execution, measurement, verification, and full-file composition
