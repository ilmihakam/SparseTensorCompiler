# IR (Intermediate Representation)

This document describes the Intermediate Representation implemented in this repository. It is split into two parts:

- **Design**: what the IR represents, why it is layered, and what semantic commitments each layer makes.
- **Implementation**: how the repository lowers AST programs into IR structures and how those structures are refined into scheduled loop trees.

The current implementation does **not** use the older `ir::Operation` / `IRProgram` model described in earlier notes. The active IR pipeline is:

```text
AST Program
  -> semantic IR
  -> scheduled IR
```

The IR layer itself is therefore best understood as three related pieces:

- a shared low-level IR utility layer in `namespace ir`
- a semantic IR in `namespace sparseir::semantic`
- a scheduled IR in `namespace sparseir::scheduled`

---

## Part 1: Design

### 1. Why the IR is layered

The source DSL expresses tensor computations in index notation, but that syntax is not yet a good form for sparse compilation. The compiler needs an internal representation that can answer three different questions:

1. What does the program mean?
2. Which indices are free, reduced, dense, or sparse?
3. What concrete loop structure should implement that meaning for CSR and CSC data?

One representation is not ideal for all three questions at once. The repository therefore separates the IR into two main layers.

- **Semantic IR** preserves the meaning of the program while making tensor accesses, iterator structure, and output behaviour explicit.
- **Scheduled IR** turns that semantic description into an executable loop tree with explicit dense loops, sparse loops, merge semantics, temporaries, and region binding information.

This split is important for sparse compilation because semantic properties and scheduling properties are related but not identical. For example, "index `k` is a reduction index" is a semantic fact, whereas "index `k` is implemented as a sparse loop driven by tensor `A` under parent index `i`" is a scheduling decision.

### 2. The three IR pieces

#### 2.1 Shared low-level IR (`namespace ir`)

The `ir` namespace is a shared support layer used by the rest of the pipeline. It provides:

- tensor metadata: `Format`, `Tensor`
- expression metadata: `RootOpKind`, `ExpressionInfo`
- sparse/output metadata: `MergeStrategy`, `OutputStrategy`
- optimization bookkeeping: `LoopOptimizations`
- structured low-level expressions/statements: `IRExpr` and `IRStmt`

This layer is not the main program IR. Instead, it provides common building blocks that the scheduled IR reuses for loop bodies, temporaries, and structured statements such as:

- `IRTensorAccess`
- `IRConstant`
- `IRBinaryOp`
- `IRScalarVar`
- `IRFuncCall`
- `IRIndexedAccess`
- `IRCompareExpr`
- `IRScalarDecl`
- `IRAssign`
- `IRVarDecl`
- `IRFreeStmt`
- `IRIfStmt`
- `IRForStmt`
- `IRRawStmt`

The presence of both structured nodes and `IRRawStmt` reflects the current state of the implementation: most lowering now builds structured statements, but some transformation paths still fall back to raw rendered C fragments.

#### 2.2 Semantic IR (`namespace sparseir::semantic`)

Semantic IR is the first compiler-owned representation after the AST. It answers:

- what tensors participate in a computation
- which indices are free and which are reduction indices
- which iterators are dense or sparse
- what merge behaviour is implied by the RHS expression
- what output strategy is required

The semantic layer uses its own expression tree:

- `Expr`
- `TensorRead`
- `Constant`
- `ScalarRef`
- `BinaryExpr`
- `CallExpr`

Program structure is represented by:

- `semantic::Declaration`
- `semantic::Call`
- `semantic::Compute`
- `semantic::Region`
- `semantic::Program`

The key node is `semantic::Compute`. It stores:

- the LHS tensor use
- the output tensor metadata
- the input tensor metadata
- the RHS semantic expression
- an `IteratorGraph`
- free and reduction index lists
- `ExpressionInfo`
- `OutputStrategy`

The semantic layer therefore preserves the meaning of the computation while making sparse-compiler analysis explicit.

#### 2.3 Scheduled IR (`namespace sparseir::scheduled`)

Scheduled IR is the executable IR of the compiler. It keeps the same program structure as semantic IR, but each compute statement is refined into a loop tree.

Program structure is represented by:

- `scheduled::Declaration`
- `scheduled::Call`
- `scheduled::Compute`
- `scheduled::Region`
- `scheduled::Program`

The core scheduling structure is `scheduled::Loop`, which contains:

- `indexName`, `lower`, `upper`
- `runtimeBound`
- `LoopKind` (`Dense`, `Sparse`, `Block`)
- `driverTensor`
- `parentIndexOverride`
- `mergeStrategy`
- `mergedTensors`
- `children`
- `preStmts`
- `postStmts`
- `isExternallyBound`
- `tileBlockSize`

`scheduled::Compute` then packages:

- the original computation metadata
- the scheduled `rootLoop`
- `outputPattern`
- `patternSources`
- `prologueStmts`
- `epilogueStmts`
- `LoopOptimizations`
- `fullyLowered`

At this point the compiler no longer reasons only about "a tensor expression"; it reasons about a concrete sparse traversal strategy.

### 3. Semantic model captured by the IR

#### 3.1 Tensor metadata

The IR uses `ir::Tensor` to record:

- tensor name
- storage format (`Dense`, `CSR`, `CSC`)
- dimensions
- source-level index order

This is enough to connect index notation to format-aware iteration. The IR does not store sparse pointer arrays directly; instead it records the information needed to decide how those arrays must be traversed later.

#### 3.2 Free and reduction indices

For a computation such as:

```text
compute C[i, j] = A[i, k] * B[k, j];
```

the IR classifies indices into:

- **free indices**: indices appearing on the LHS (`i`, `j`)
- **reduction indices**: indices that appear on the RHS but not the LHS (`k`)

This classification is central. It determines iterator order, reduction behaviour, output shape, and which loops will carry accumulation.

#### 3.3 Iterator graph

The semantic IR makes iteration structure explicit using:

- `IteratorSource`
- `IteratorNode`
- `IteratorGraph`

Each `IteratorNode` represents one logical index. It records:

- the index name
- its inferred upper bound
- whether it is a reduction index
- whether it behaves as a dense or sparse iterator
- whether multiple sparse sources imply `Union` or `Intersection`
- which tensors contribute that iterator

This is the semantic bridge between index notation and loop scheduling. It says, for example, "index `j` is sparse because it is the sparse access position of a CSR tensor" before the scheduler decides how to materialise that fact as loops.

#### 3.4 Output strategy vs output pattern

The IR separates two related ideas.

- `ir::OutputStrategy` answers how the result is represented:
  - `DenseArray`
  - `SparseFixedPattern`
  - `HashPerRow`
- `sparseir::OutputPatternKind` answers what sparse pattern semantics the scheduled compute follows:
  - `None`
  - `Union`
  - `Intersection`
  - `Sampled`
  - `DynamicRowAccumulator`

This distinction matters because sparse outputs are not all the same.

- **Pointwise sparse union**: sparse-sparse addition
- **Pointwise sparse intersection**: sparse elementwise multiplication
- **Sampled output**: sparse mask multiplied by a dense contraction, as in SDDMM
- **Dynamic sparse accumulation**: sparse output built through workspace accumulation, as in sparse-output SpGEMM

### 4. What scheduling adds

Scheduling refines a semantic compute into a loop tree with explicit implementation choices.

#### 4.1 Loop kinds

The scheduled IR uses three loop kinds.

- `Dense`: a standard affine loop over a bounded integer range
- `Sparse`: a loop driven by sparse structure from a tensor in CSR or CSC format
- `Block`: a strip-mining wrapper introduced by later loop-tree transforms

The semantic layer may know that an index is sparse; the scheduled layer decides which tensor drives it and under which parent index it is traversed.

#### 4.2 Sparse parent-child relationships

Sparse traversal in CSR and CSC is hierarchical. A sparse iterator often depends on an outer coordinate.

For this reason the scheduled loop stores:

- `driverTensor`: which tensor provides the sparse structure
- `parentIndexOverride`: which outer index should be used as the sparse parent when the default lexical parent is not correct

This is especially important for CSC scheduling, where the natural sparse parent-child relationship differs from CSR.

#### 4.3 Structured loop bodies

The scheduled loop tree does not just store nesting. It can also store structured actions around the loop:

- `preStmts` for actions before child loops
- `postStmts` for actions after child loops

This is how the IR represents:

- scalar accumulator initialisation for sampled contractions
- final writes into sparse output value arrays
- workspace setup inside outer loops
- gather-and-clear loops for dynamic sparse accumulation

This is a major improvement over a pure string-template approach because the IR can still be transformed while preserving structure.

#### 4.4 Regions and external binding

The DSL supports explicit `for [tensors] [indices] { ... }` regions. The IR models these as `semantic::Region` and `scheduled::Region`.

Region nodes record:

- the tensor list
- the region index list
- per-index runtime bounds
- nested statements

When a scheduled compute is nested under a region, scheduling marks matching leading loops as `isExternallyBound = true`. This means those loops are already supplied by the enclosing region and should be treated as bound by context rather than emitted again.

### 5. Worked examples

#### 5.1 CSR SpMV

Source:

```text
compute y[i] = A[i, j] * x[j];
```

Semantic facts:

- free indices: `i`
- reduction indices: `j`
- `A` is sparse, `x` is dense
- output strategy: `DenseArray`
- iterator graph: `i` is dense, `j` is sparse

Scheduled shape:

```text
Loop i  (dense)
  Loop j  (sparse, driven by A)
    y[i] += A_vals[pA] * x[j]
```

This is the canonical sparse row traversal for CSR.

#### 5.2 CSC SpMM

Source:

```text
compute C[i, j] = A[i, k] * B[k, j];
```

with `A` stored as CSC.

Semantic facts:

- free indices: `i`, `j`
- reduction index: `k`
- one sparse input, one dense input

Scheduled shape:

```text
Loop k  (dense)
  Loop i  (sparse, driven by A, parent = k)
    Loop j  (dense)
      C[i, j] += ...
```

The important detail is that the sparse loop over `i` uses `parentIndexOverride = "k"`. This expresses that CSC storage is column-oriented, so the sparse traversal must be anchored on the column index rather than the default lexical parent.

#### 5.3 Sparse-output SpGEMM

Source:

```text
compute C[i, j] = A[i, k] * B[k, j];
```

with sparse `A`, sparse `B`, and sparse output `C`.

Semantic facts:

- two sparse inputs
- one reduction index
- sparse output cannot be treated as a preallocated dense array
- output strategy: `HashPerRow`
- output pattern: `DynamicRowAccumulator`

Scheduled consequences:

- `prologueStmts` allocate workspace arrays such as accumulator and marking buffers
- the outer loop owns workspace setup/teardown logic
- inner sparse loops update the workspace rather than writing directly to `C`
- `postStmts` contain structured gather-and-clear loops that materialise the row pattern

This example shows why the scheduled IR must represent more than simple nested loops: sparse output assembly requires temporaries, ownership, and structured post-processing.

---

## Part 2: Implementation

### 6. Where the IR is implemented

The active IR implementation is split across:

- `include/ir.h`
- `include/semantic_ir.h`
- `src/ir.cpp`
- `src/semantic_ir.cpp`

Conceptually:

- `ir.h` defines shared metadata and low-level structured statements/expressions
- `semantic_ir.h` defines semantic and scheduled IR data structures
- `semantic_ir.cpp` implements lowering from AST to semantic IR, then scheduling to scheduled IR
- `ir.cpp` implements cloning and rendering helpers for the shared low-level `ir` nodes

### 7. AST to semantic IR

The first lowering stage is `lowerToSemanticProgram(...)`.

Its implementation performs four main tasks.

#### 7.1 Collect declarations

The compiler first walks the AST and collects tensor declarations into a map from tensor name to `ir::Tensor`. During this pass it:

- converts format strings such as `"CSR"` and `"CSC"` into `ir::Format`
- parses declared shapes into integer dimensions
- recurses through nested `for` regions so nested computes can resolve tensor metadata

This gives later stages a uniform source of tensor format and shape information.

#### 7.2 Lower expressions

AST expressions are lowered into semantic expressions:

- tensor access -> `TensorRead`
- number -> `Constant`
- identifier -> `ScalarRef`
- binary operator -> `BinaryExpr`
- function call -> `CallExpr`

At this stage the structure of the original expression is preserved, but the compiler no longer depends on parser-specific AST classes.

#### 7.3 Build `semantic::Compute`

For each compute statement, the compiler constructs a `semantic::Compute` and fills:

- `lhs`
- `output`
- `inputs`
- `rhs`
- `freeIndices`
- `reductionIndices`
- `exprInfo`
- `outputStrategy`

`exprInfo` is derived by analysing the RHS expression:

- root operation kind (`ADD` or `MULT`)
- whether the RHS is wrapped in a function call
- how many tensor reads occur
- how many inputs are sparse or dense
- which tensors belong to each category

This stage is also where input tensors are deduplicated: if a tensor is read multiple times in the RHS, it still appears once in `inputs`.

#### 7.4 Build the iterator graph

Once tensor accesses and indices are known, the compiler builds an `IteratorGraph`.

For each logical index it:

- infers an upper bound
- records whether the index is free or reduction
- records every tensor/source position where that index appears
- checks whether that position is sparse for the tensor format
- classifies the iterator as dense or sparse
- assigns `MergeKind::Union` or `MergeKind::Intersection` when multiple sparse sources share the same iterator

This graph is the semantic summary that drives scheduling.

### 8. Semantic IR program structure

`lowerToSemanticProgram(...)` preserves full program structure, not just isolated kernels.

- declarations become `semantic::Declaration`
- calls become `semantic::Call`
- compute statements become `semantic::Compute`
- `for` statements become `semantic::Region`

For each region, the compiler also computes `runtimeBounds` strings for the region indices. These are derived from the participating tensors and are carried into scheduled IR unchanged.

### 9. Semantic to scheduled IR

The second lowering stage is `scheduleProgram(...)` and `scheduleCompute(...)`.

This stage converts iterator semantics into an explicit loop tree.

#### 9.1 Derive loop lowering specifications

The scheduler first computes an ordered iterator list:

- all free indices first
- all reduction indices after them

For each index it then derives a `LoopLoweringSpec` containing:

- loop kind
- upper bound
- merge strategy
- merged tensor list
- driver tensor
- parent index
- accumulation strategy
- workspace ownership metadata when needed

This step is where the semantic iterator graph becomes executable loop policy.

#### 9.2 Select output-aware lowering behaviour

The scheduler treats several cases specially.

- **Dense output**: leaf loops write directly to the destination tensor, using accumulation when reduction indices are present.
- **Sparse fixed-pattern output**: sparse loops are driven by the output tensor pattern, and loop bodies fill output value slots rather than re-discovering coordinates.
- **Sampled dense contraction**: the scheduler introduces a scalar accumulator such as `sum`, attaches initialisation to `preStmts`, attaches reduction updates to an inner loop, and emits the final sampled write in `postStmts`.
- **Dynamic row accumulation**: the scheduler allocates workspace through `prologueStmts`, performs structured updates in inner sparse loops, and emits gather/clear loops in `postStmts`.

These cases are why scheduled IR stores both loop structure and structured statements.

#### 9.3 Build the loop tree

`buildGenericLoopNest(...)` recursively constructs `scheduled::Loop` nodes from the derived specs.

During construction it sets:

- `LoopKind`
- `driverTensor`
- `parentIndexOverride`
- `mergeStrategy`
- `mergedTensors`
- `runtimeBound`
- `preStmts`
- `postStmts`

Leaf behaviour depends on the output strategy and whether random sparse access is required. When needed, the lower-level `ir::IRTensorAccess` node can be marked with `useRandomAccess` and the correct helper name such as `sp_csr_get` or `sp_csc_get`.

If loop construction succeeds, `scheduled::Compute::fullyLowered` is set to `true`.

### 10. Scheduling regions and external binding

Scheduling a region is not just recursive scheduling of its body. It also threads enclosing index information downward.

The scheduler:

- accumulates enclosing region indices
- schedules nested statements
- checks whether the leading scheduled loops match those enclosing indices
- marks matching loops as externally bound

This is implemented by `markExternallyBound(...)`. Its effect is to encode, inside the IR itself, that some loop variables come from the surrounding program context rather than from the compute's own emitted loop headers.

### 11. Low-level structured IR inside scheduled IR

Although the semantic and scheduled layers are the main IRs, the shared `ir` nodes are an important implementation detail.

They are used for:

- loop-local scalar declarations
- structured assignments
- temporary workspace variables
- conditional updates
- helper loops inside post-processing
- random-access sparse reads

This makes the scheduled IR richer than a plain loop header tree. It can carry a whole structured micro-program at each loop boundary.

### 12. IR fields that exist for later scheduling transforms

Some scheduled-IR fields exist because later scheduling transforms may rewrite the loop tree.

Within the IR layer itself, the important point is only that these fields are part of the representation:

- `LoopKind::Block`
- `tileBlockSize`
- `LoopOptimizations`

They should be understood as annotations and structural extensions on top of scheduled IR, not as a separate IR stage.

### 13. Current status of the IR layer

The implemented IR layer has three clear properties.

First, the semantic and scheduled split is real and complete: semantic IR captures meaning, while scheduled IR captures executable sparse traversal.

Second, the current compiler is no longer organised around a monolithic kernel node such as `ir::Operation`. The active kernel representation is `scheduled::Compute` plus its loop tree.

Third, the IR is already rich enough to represent:

- format-aware CSR and CSC traversal
- sparse merge semantics
- fixed-pattern sparse outputs
- sampled contractions
- dynamic sparse output assembly
- region-scoped externally bound loops

That makes it an appropriate substrate for the compiler's current research focus: sparse loop scheduling on a representation that still preserves the meaning of the original index notation.
