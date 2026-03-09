# Optimisations.md — SparseTensorCompiler Optimization System

This document covers the optimization system from two angles: **Design** (goals, types, legality conditions, strategies) and **Implementation** (transformation passes, pattern detection, algorithms, IR rewriting, pass ordering).

---

## Part 1: Design

### 1. Overview & Motivation

The compiler applies optimizations at the **IR level**, operating on `ir::Operation` loop trees after AST lowering and before code generation. This placement is deliberate:

- **Not source-level**: The DSL captures index notation semantics only; it carries no schedule information.
- **Not codegen-level**: Code generation is a mechanical translation; embedding optimization logic there would make it untestable and non-composable.
- **IR-level**: The `Loop` tree is a clean, mutable data structure. Passes read and write it in isolation. Each pass is independently testable, and their composition is controlled by a single switch.

The **research goal** is to characterize when loop interchange and loop blocking actually improve performance on real sparse matrices, and derive heuristics that predict which optimization to apply based on input features (density, clustering, storage format). Generating multiple schedules from the same `.tc` input — and comparing them on the same matrix — is the core empirical method.

The pipeline position of optimizations:

```
AST (Program)
   ↓  lowerToIR()
ir::Operation  { rootLoop, inputs, output, kernelType, optimizations }
   ↓  applyOptimizations()    ← all three passes run here
   ↓  generateCode()
standalone C file
```

The three passes form a fixed taxonomy:
1. **Format-correctness reordering** — a correctness pass; always runs unconditionally.
2. **Loop interchange** — a performance pass; user-controlled via `--opt-interchange`.
3. **Loop blocking / tiling** — a performance pass; user-controlled via `--opt-block=SIZE`.

---

### 2. Optimization Goals

**Cache locality** is the primary target. Sparse kernels like SpMV and SpMM are memory-bound: the bottleneck is the rate at which data is fetched from DRAM into L1/L2 cache, not arithmetic throughput.

**Spatial locality** means accessing memory sequentially so that an entire cache line (typically 64 bytes, holding 8 `double` values) is useful when loaded. CSR natural traversal streams through `vals[]` and `col_idx[]` contiguously — this is the ideal pattern for spatial locality.

**Temporal locality** means reusing data already in cache before eviction. For SpMV, the vector `x` is accessed at `x[col_idx[p]]` for each nonzero. If the column indices of row `i` span a wide range, different entries of `x` are accessed, potentially evicting earlier entries before they are reused in subsequent rows. Loop blocking groups rows into tiles, concentrating the `x` accesses of each tile into a smaller working set that fits in L1.

**Memory bandwidth reduction** follows from better locality: fewer cache misses means fewer trips to DRAM. For large matrices where `x` does not fit in L1 (L1 is typically 32–64 KB; a vector of 8,000 doubles is 64 KB), blocking with size 32 keeps the tile's working set small enough to fit.

Connection to concrete hardware numbers:
- L1 data cache: 32–64 KB (fits ~4,000–8,000 doubles)
- L2 cache: 256 KB–1 MB
- L3 / LLC: 4–32 MB
- DRAM latency: ~100 ns vs L1 latency ~4 cycles

A block size of 32 rows means the `x` entries touched by those rows form a working set whose size depends on column span, not matrix dimension. For matrices with clustered nonzeros, this working set is small and fits in L1.

---

### 3. Optimization Types

#### 3.1 Format-Correctness Reordering (automatic, not user-controlled)

**Purpose**: Align the loop iteration order with the tensor storage format so that the inner loop streams through contiguous memory.

CSR stores nonzeros row-by-row. The natural traversal is:
```c
for (int i = 0; i < M; i++)                          // outer: row
    for (int p = row_ptr[i]; p < row_ptr[i+1]; p++)  // inner: column pointer
```

CSC stores nonzeros column-by-column. The natural traversal is:
```c
for (int j = 0; j < N; j++)                          // outer: column
    for (int p = col_ptr[j]; p < col_ptr[j+1]; p++)  // inner: row pointer
```

If the DSL index expression inverts this order — e.g., `A[j, i]` with `A : CSR` — the IR lowering would produce a column-outer loop on a row-stored matrix, causing random jumps through `row_ptr` and breaking spatial locality. The reordering pass detects this mismatch and swaps the loop nesting to restore format alignment.

This pass runs unconditionally before any user-controlled optimization. It is a correctness guarantee, not a toggle.

#### 3.2 Loop Interchange (`--opt-interchange`)

**Purpose**: Reorder loop nesting to improve the locality of output writes or intermediate accumulations.

For SpMM, the standard IR after lowering produces:
```
Loop i (dense, 0..M)
  Loop k (sparse, CSR over row i)
    Loop j (dense, 0..N)
      C[i][j] += A_vals[p] * B[k][j]
```

The innermost loop writes to `C[i][j]` with `j` varying — this is already contiguous. However, the write to `C[i][j]` occurs inside the sparse `k` loop, meaning each `C[i][j]` cell is touched once per nonzero in row `i`. Interchange produces:
```
Loop i (dense, 0..M)
  Loop j (dense, 0..N)
    Loop k (sparse, CSR over row i)
      C[i][j] += A_vals[p] * B[k][j]
```

Now `C[i][j]` is accumulated across all nonzeros of row `i` before moving to the next `j` — but the key benefit is that `C[i][j]` is kept as a scalar accumulator in a register across the `k` loop, eliminating the repeated load/store pattern.

Loop interchange acts on the loop tree by swapping the `index` and `kind` fields of two adjacent `Loop` nodes without restructuring the tree itself (children pointers remain the same).

#### 3.3 Loop Blocking / Tiling (`--opt-block=SIZE`)

**Purpose**: Strip-mine an iteration space to create a tile whose working set fits in a target cache level.

For SpMV, blocking the row loop:

**Before** (block size 32):
```c
for (int i = 0; i < M; i++) {
    for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
        y[i] += vals[p] * x[col_idx[p]];
    }
}
```

**After**:
```c
for (int i_block = 0; i_block < (M + 31) / 32; i_block++) {
    int i_start = i_block * 32;
    int i_end = (i_start + 32 < M) ? i_start + 32 : M;
    for (int i = i_start; i < i_end; i++) {
        for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
            y[i] += vals[p] * x[col_idx[p]];
        }
    }
}
```

Within each tile of 32 rows, the column indices accessed are limited. If the matrix has clustered nonzeros, the set of distinct `x[j]` entries accessed per tile is small — potentially fitting in L1 — meaning the next tile's accesses hit warm cache lines rather than cold ones.

The blocking pass inserts a new `Loop` node (`i_block`) above the target loop as a wrapper. The target loop remains structurally unchanged; only the outer block loop is new. The code generator recognizes `tileBlockSize != 0` in the block loop and emits the `i_start`/`i_end` bounds logic.

A 2D variant (`--opt-block2d=ROWS,COLS`) tiles two indices simultaneously, typically both `i` and `j` for SpMM. This creates a 2D tile where both the output row slice and the output column slice are bounded.

#### 3.4 Available Schedule Configurations

| Config | CLI flags | `OptConfig` factory method |
|--------|-----------|---------------------------|
| Baseline | _(none)_ | `OptConfig::baseline()` |
| Interchange only | `--opt-interchange` | `OptConfig::interchangeOnly()` |
| Blocking only | `--opt-block=32` | `OptConfig::blockingOnly(32)` |
| Both, interchange first | `--opt-all=32` | `OptConfig::allOptimizations(32, I_THEN_B)` |
| Both, blocking first | `--opt-all=32 --opt-order=B_THEN_I` | `OptConfig::allOptimizations(32, B_THEN_I)` |
| Both, double interchange | `--opt-all=32 --opt-order=I_B_I` | `OptConfig::allOptimizations(32, I_B_I)` |
| 2D blocking | `--opt-block2d=32,32` | `OptConfig::blocking2D(32, 32)` |

---

### 4. Where Optimization Occurs

All optimization passes are **pure mutations of `ir::Operation`**. They read and write the `rootLoop` tree and the `optimizations` metadata struct. They do not touch:
- The AST or DSL representation
- The lexer/parser
- The code generator's emission logic (beyond relying on metadata flags it reads)

The `LoopOptimizations` struct inside `ir::Operation` serves as the communication channel between passes and the code generator:

```cpp
struct LoopOptimizations {
    bool reorderingApplied = false;
    bool interchangeApplied = false;
    bool blockingApplied = false;
    int blockSize = 32;
    std::string tiledIndex;          // e.g., "i" or "j"
    bool blocking2DApplied = false;
    std::vector<std::string> tiledIndices;
    std::vector<int> blockSizes;
    // ... original/new order for reordering recording
};
```

The code generator reads `blockingApplied`, `blockSize`, `tiledIndex` etc. to emit the correct `// OPTS:` comment header and to confirm the IR structure it is about to traverse.

---

### 5. Legality Conditions

A transformation is legal if it produces a result numerically equivalent to the untransformed program for all valid inputs.

#### 5.1 Format-Correctness Reordering

Always legal for 2-deep loop nests when:
- The outer loop is dense (no sparse pointer dependency)
- The inner loop is sparse over a tensor whose format is being corrected

Current limitation: the pass is restricted to 2-deep SpMV nests. Applying it to 3-deep SpMM nests requires verifying that the middle loop's body references are still valid after the swap — this is marked as a TODO.

#### 5.2 Loop Interchange

Interchange is legal when the inner loop's iteration set does not depend on the outer loop's iteration variable being moved inward. The check is performed by `isInterchangeLegal(outer, middle, inner)`:

| Outer | Middle | Inner | Legal? | Reason |
|-------|--------|-------|--------|--------|
| Dense | Sparse | Dense | Yes | Dense inner has no dependency on sparse middle's extracted index |
| Dense | Dense | Sparse | Yes | Swapping moves sparse inward; outer scope still provides all needed variables |
| Dense | Sparse | Sparse | **No** | Inner sparse iterates `B->row_ptr[k]` where `k = A->col_idx[pA]`; swapping would break this data dependency |
| Dense | Dense | Dense | Yes | No sparse dependencies; semantics are permutation-invariant |

The Dense → Sparse → Sparse case arises in SpGEMM (sparse times sparse), which is not a primary target but is guarded against.

An additional check: a block wrapper loop (`i_block`) cannot be promoted to inner — `isBlockLoopIndex()` detects names containing `"_block"` and prevents interchange from treating them as candidates for the `inner` role.

#### 5.3 Loop Blocking

Blocking is always legal when applied to a `Dense` loop. Splitting `for i in [0, M)` into `for i_block` and bounded `for i` is a pure range decomposition that covers the same set of iterations in the same order. The output values are identical.

Guards:
- Only `LoopKind::Dense` loops are tiled; `LoopKind::Sparse` loops cannot be tiled (the bounds `row_ptr[i]..row_ptr[i+1]` depend on `i` and cannot be split independently).
- The `blockingApplied` flag prevents double-blocking (idempotency guard).

---

### 6. Optimization Strategies

**When to use interchange**:
- Primary case: SpMM with large output matrix `C`. Interchange improves `C[i][j]` write locality by accumulating the full `k` sum before moving to the next `j`, keeping the accumulator in a register.
- Most effective when rows have many nonzeros (high average `nnz_per_row`), since the `k` loop runs long and the register accumulation benefit is proportional.
- Less effective for very sparse matrices (few nonzeros per row), where the `k` loop is short and the overhead of the interchange restructuring outweighs the benefit.

**When to use blocking**:
- Primary case: SpMV with large input vector `x`. Blocking the row loop creates tiles whose column-index accesses concentrate into a smaller range of `x`, increasing L1 reuse.
- Benefit scales with column-index clustering: matrices where nearby rows access nearby columns (low column span variance) gain the most from blocking.
- Flat or random column index distributions see minimal benefit because the `x` working set per tile is still large.

**When to use I→B (default) vs B→I vs I→B→I**:

`I_THEN_B` (default): interchange reorders the logical loop structure first, producing a clean loop nest, then blocking tiles it. The tile boundaries align with the interchanged structure. This is the compositionally clean default.

`B_THEN_I`: blocking tiles the loop first, then interchange runs on the tiled loop tree. The interchange pass in this mode sets `allowBlockWrappedDenseInner = true`, enabling it to look through `_block` wrapper loops and apply interchange at a deeper position. This lets interchange operate on the inner tile loop rather than being blocked by the wrapper.

`I_B_I`: a double interchange with blocking between. The first interchange reshapes the loop, blocking tiles it, and the second interchange re-applies on the tiled structure. This explores a schedule composition not reachable by a single interchange. It is primarily a research configuration for comparing schedule spaces.

---

## Part 2: Implementation

### 7. IR Data Structures Used by Passes

The passes operate on `ir::Loop` nodes. Key fields:

| Field | Type | Role |
|-------|------|------|
| `index.name` | `string` | Loop variable name, e.g., `"i"`, `"j_block"` |
| `index.lower` | `int` | Loop lower bound (typically 0) |
| `index.upper` | `int` | Loop upper bound (dimension size) |
| `index.isSparse` | `bool` | Whether this index iterates over nonzeros |
| `kind` | `LoopKind` | `Dense` or `Sparse` — determines codegen pattern |
| `children` | `vector<unique_ptr<Loop>>` | Nested child loops |
| `body` | `string` | Innermost C statement (leaf node only) |
| `preBody` | `string` | Emitted before children, e.g., accumulator init `"double sum = 0.0;"` |
| `sparseTensorName` | `string` | Which tensor drives the sparse iteration (for CSR/CSC pointer variable naming) |
| `tileBlockSize` | `int` | Non-zero signals this is a block wrapper loop; codegen emits `i_start`/`i_end` |
| `isExternallyBound` | `bool` | Skip loop header emission; loop variable is already bound by enclosing scope |

The `ir::Operation` contains the loop tree (`rootLoop`) and the optimization metadata:

```cpp
struct Operation {
    std::string kernelType;              // "spmv" or "spmm"
    std::vector<Tensor> inputs;
    Tensor output;
    std::unique_ptr<Loop> rootLoop;
    LoopOptimizations optimizations;
};
```

`LoopOptimizations` is both a record of what was applied and a signal to the code generator:

```cpp
struct LoopOptimizations {
    bool reorderingApplied = false;
    std::vector<std::string> originalOrder;   // e.g., {"j", "i"}
    std::vector<std::string> newOrder;        // e.g., {"i", "j"}

    bool interchangeApplied = false;

    bool blockingApplied = false;
    int blockSize = 32;
    std::string tiledIndex;                   // e.g., "i" or "j"

    bool blocking2DApplied = false;
    std::vector<std::string> tiledIndices;    // e.g., {"i", "j"}
    std::vector<int> blockSizes;              // per-index block sizes
};
```

---

### 8. Format Analysis Helper Functions (`src/optimizations.cpp`)

#### `getNaturalOrder(Format fmt)`

Returns the preferred index role sequence for a storage format:
- CSR, Dense → `{"row", "col"}`
- CSC → `{"col", "row"}`

This is used indirectly via `computeNaturalOrder`.

#### `needsReordering(const ir::Tensor& t)`

Determines whether a tensor's DSL index order conflicts with its storage format:
- Dense tensors: always returns `false` (row-major works either way)
- Tensors with fewer than 2 indices: returns `false`
- CSR: returns `true` if `indices[1] < indices[0]` alphabetically (i.e., the second index name sorts before the first, indicating column-first access on a row-stored format)
- CSC: returns `true` if `indices[0] < indices[1]` alphabetically (i.e., the first index sorts before the second, indicating row-first access on a column-stored format)

**Known limitation**: The alphabetical heuristic works correctly only for conventional index naming (`i`, `j`, `k`). Non-conventional names may produce incorrect results.

#### `computeNaturalOrder(const ir::Tensor& t)`

Returns the index names in the order the format wants them:
- CSR: alphabetically earlier index first (treated as row index)
- CSC: alphabetically later index first (treated as column index)
- Dense: original order unchanged

#### `collectLoopOrder(const ir::Loop* loop)`

Traverses the loop tree depth-first (following the first child at each level) and returns the sequence of `index.name` values encountered. This represents the current loop nesting order as a flat list, e.g., `{"i", "j"}` for a 2-deep nest.

---

### 9. Pattern Detection

Before any structural mutation, the passes use pattern detectors to confirm applicability.

#### `canSwapLoops(const ir::Loop* root)`

Used by the reordering pass. Returns `false` if the child loop is `Sparse` (promoting a sparse loop to outer position would change which tensor drives the pointer bounds). Returns `true` only for Dense-outer → Dense-inner patterns.

#### `isInterchangeLegal(const Loop* outer, const Loop* middle, const Loop* inner)`

Used by the interchange pass before any mutation. Implements the legality table from Section 5.2:

```
if middle->kind == Sparse && inner->kind == Sparse → return false  (SpGEMM; data dependency)
else → return true
```

#### `isBlockLoopIndex(const std::string& name)`

Returns `true` if the name contains `"_block"`. Prevents the interchange pass from treating a block wrapper loop as a candidate for the inner role in a swap.

#### `fuseAccumulatorPattern(preBody, innerBody, postBody)`

Detects the three-part accumulator pattern that arises in sparse loops:
- `preBody` contains `"double sum = 0.0;"`
- `innerBody` contains `"sum += <expr>"`
- `postBody` references `"sum"` in a write like `"y[i] = sum;"`

If detected, extracts `<expr>` and rewrites `postBody`'s assignment from `"= sum"` to `"+= <expr>"`, eliminating the scalar intermediate. Returns `false` if the pattern is not present, in which case the interchange falls back to string-based body propagation.

---

### 10. Loop Transformation Algorithms

#### `swapRootChildLoops(ir::Loop* root)` — reordering

Swaps the `index` and `kind` fields between a root loop and its first child, without changing the tree structure:

```
Input:  Loop(index=A, kind=Dense)
          └── Loop(index=B, kind=Dense)

Output: Loop(index=B, kind=Dense)
          └── Loop(index=A, kind=Dense)

Algorithm:
  std::swap(root->index, child->index);
  std::swap(root->kind,  child->kind);
```

The children pointers are unchanged; only the loop headers are swapped. This is safe because the loop body at the leaf does not reference the index names directly in the IR — names are resolved during code generation.

#### `blockLoopByIndex(ir::Operation& op, int blockSize, const std::string& targetIndex)` — blocking

Inserts a block wrapper loop above the target loop in the tree:

```
Input:  ... → Loop(name="i", kind=Dense, upper=M) → ...

Output: ... → Loop(name="i_block", kind=Dense, upper=ceil(M/B), tileBlockSize=B)
                └── Loop(name="i", kind=Dense, upper=M) → ...

Algorithm:
  1. Find target loop by DFS on name == targetIndex
  2. Create blockLoop:
       blockLoop.index.name  = targetIndex + "_block"
       blockLoop.index.upper = (M + blockSize - 1) / blockSize
       blockLoop.tileBlockSize = blockSize
  3. blockLoop.children = { targetLoop }
  4. Replace targetLoop's slot in its parent with blockLoop
  5. Update op.optimizations metadata
```

The code generator, on encountering a loop with `tileBlockSize != 0`, emits:
```c
int i_start = i_block * B;
int i_end   = (i_start + B < M) ? i_start + B : M;
```
and uses `i_start`/`i_end` as the bounds for the inner `i` loop.

#### `tryInterchangeAtDenseNode(ir::Loop* node, bool allowBlockWrappedDenseInner)` — interchange

Two cases are handled:

**Forward case (Dense → Sparse → Dense):**
```
Input:  node(Dense)
          └── middle(Sparse, preBody="double sum = 0.0;", body="y[i] = sum;")
                └── inner(Dense, body="sum += vals[p] * x[j];")

Output: node(Dense)
          └── newDense(Dense, body="")
                └── newSparse(Sparse, body="y[i] += vals[p] * x[j];")

Steps:
  1. Check isInterchangeLegal(node, middle, inner)
  2. Attempt fuseAccumulatorPattern(middle->preBody, inner->body, middle->body)
  3. If fused: newSparse.body = fusedBody; newSparse.preBody = ""
     Else:     newSparse.body = inner->body
  4. Build newDense: copy dense metadata from inner; children = [newSparse]
  5. Replace node->children with [newDense]
```

**Reverse case (Dense → Dense → Sparse):**
```
Input:  node(Dense)
          └── middle(Dense)
                └── inner(Sparse, preBody, body="C[i][j] += ...")

Output: node(Dense)
          └── newSparse(Sparse, preBody)
                └── newDense(Dense, body="C[i][j] += ...")

Steps:
  1. Check isInterchangeLegal(node, middle, inner)
  2. Build newDense: copy dense metadata from middle; body = inner->body; children = inner->children
  3. Build newSparse: copy sparse metadata from inner; body = ""; children = [newDense]
  4. Replace node->children with [newSparse]
```

#### `applyOneInterchangeDFS(ir::Loop* node, bool allowBlockWrapped)` — interchange DFS driver

Walks the loop tree depth-first. At each node:
1. Attempt `tryInterchangeAtDenseNode(node, allowBlockWrapped)`
2. If successful, return `true` immediately (one interchange per call)
3. If not, recurse into each child

This ensures at most one interchange per pass invocation. Multiple passes (e.g., `I_B_I`) call this multiple times.

---

### 11. Pass Implementations

#### `applyReordering(ir::Operation& op)` (`src/optimizations.cpp`)

```
1. Guard: if reorderingApplied → return early (idempotency)
2. sparseTensor = findSparseTensor(op.inputs)  [first CSR or CSC tensor]
3. currentOrder = collectLoopOrder(op.rootLoop)
4. desiredOrder = computeNaturalOrder(*sparseTensor)
5. If currentOrder != desiredOrder
   AND sizes match
   AND it is a 2-deep inversion:
     if canSwapLoops(rootLoop): swapRootChildLoops(rootLoop)
6. Record originalOrder = currentOrder
         newOrder     = desiredOrder
         reorderingApplied = true
```

The 2-deep inversion check ensures the pass does not attempt to reorder 3-deep or partially-matching nests, which would require more complex analysis.

#### `applyBlocking(ir::Operation& op, const OptConfig& config)` (`src/optimizations.cpp`)

```
1. Guard: if !enableBlocking OR blockingApplied → return
2. Kernel-specific target selection:
   - "spmm": tile "j" (or both "i" and "j" for 2D config)
   - "sddmm": tile "k" (or both "i" and "k" for 2D config)
   - others: tile outermost dense loop (found by traversal)
3. For each target index: blockLoopByIndex(op, blockSize, targetIndex)
4. Update metadata: blockingApplied = true; blockSize; tiledIndex
```

#### `applyLoopInterchange(ir::Operation& op, const OptConfig& config)` (`src/optimizations.cpp`)

```
1. Guard: if !enableInterchange → return
2. Guard: if rootLoop has no children → return
3. allowBlockWrapped = (config.order == B_THEN_I && config.enableBlocking)
4. success = applyOneInterchangeDFS(op.rootLoop, allowBlockWrapped)
5. If success: op.optimizations.interchangeApplied = true
```

The `allowBlockWrapped` flag is the key difference in B_THEN_I mode: it allows `tryInterchangeAtDenseNode` to descend into or across block wrapper loops, enabling interchange to operate on the inner tile structure rather than stopping at the block boundary.

---

### 12. Pass Ordering — `applyOptimizations` (`src/optimizations.cpp`)

`applyOptimizations(ir::Operation& op, const OptConfig& config)` is the single public entry point called by `main.cpp`. It enforces the ordering contract:

```cpp
// Format-correctness: always runs first, unconditionally
applyReordering(op);

// User-controlled optimizations: ordered by config.order
switch (config.order) {
    case OptOrder::I_THEN_B:
        if (config.enableInterchange) applyLoopInterchange(op, config);
        if (config.enableBlocking)    applyBlocking(op, config);
        break;

    case OptOrder::B_THEN_I:
        if (config.enableBlocking)    applyBlocking(op, config);
        if (config.enableInterchange) applyLoopInterchange(op, config);
        break;

    case OptOrder::I_B_I:
        if (config.enableInterchange) applyLoopInterchange(op, config);
        if (config.enableBlocking)    applyBlocking(op, config);
        if (config.enableInterchange) applyLoopInterchange(op, config);
        break;
}
```

Key properties of this design:
- Reordering is outside the switch and cannot be affected by `config.order`. It is always first.
- Each pass function contains its own guard (`enableInterchange`, `enableBlocking`, idempotency flags), so calling `applyLoopInterchange` when `enableInterchange = false` is a no-op.
- The `I_B_I` case calls `applyLoopInterchange` twice. Because `interchangeApplied` is not used as an idempotency guard for interchange (unlike blocking), both calls execute. The second call operates on the already-blocked loop tree, potentially finding a different interchange opportunity than the first call did.
- The switch is the single place where composition semantics are defined. Adding a new order requires only adding a new case here.

---

### 13. Generated Code Annotations

Every generated C file includes a header comment recording the active optimization configuration:

```c
// OPTS: interchange=1 block=1 block_size=32 order=I_THEN_B
```

This annotation is emitted by the code generator reading `op.optimizations` fields. It serves two purposes:
1. The generated file is self-documenting — the optimization state is embedded alongside the kernel.
2. Benchmark result CSV rows can be linked back to the source by matching the annotation.

The benchmark CSV schema records the full configuration:

```
matrix_id, kernel, interchange, block, block_size, order,
runtime_ms_min, runtime_ms_median,
cycles, instructions, L1_misses, LLC_misses, dTLB_misses
```

For TACO comparison rows, `interchange = 0`, `block = 0`, `order = "NA"`, and `impl = "taco"`.

---

### 14. `OptConfig` Structure Reference (`include/optimizations.h`)

```cpp
enum class OptOrder {
    I_THEN_B = 3,   // Interchange → Block (default)
    B_THEN_I = 4,   // Block → Interchange
    I_B_I    = 5    // Interchange → Block → Interchange
};

struct OptConfig {
    bool enableInterchange = false;
    bool enableBlocking    = false;
    int  blockSize         = 32;
    OptOrder order         = OptOrder::I_THEN_B;
    std::string outputFile = "output.c";

    // Factory methods
    static OptConfig baseline();
    static OptConfig interchangeOnly();
    static OptConfig blockingOnly(int blockSize = 32);
    static OptConfig allOptimizations(int blockSize = 32,
                                      OptOrder order = OptOrder::I_THEN_B);
    static OptConfig withBothOpts(int blockSize = 32,
                                  OptOrder order = OptOrder::I_THEN_B);
    static OptConfig blocking2D(int rowBlockSize = 32, int colBlockSize = 32);
};
```

The factory methods are the recommended way to construct configs in tests and the driver. They document intent (`baseline()`, `interchangeOnly()`) rather than requiring callers to set boolean flags manually.
