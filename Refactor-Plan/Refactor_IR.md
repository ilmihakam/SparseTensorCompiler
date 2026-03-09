Here's a **high‑level refactor design** for your sparse iteration IR that supports CSR/CSC generically, smooth optimisation passes, and correctness guarantees. This is inspired by taco's iteration graphs + MLIR sparse_tensor patterns but simplified for your scope. [commit.csail.mit](https://commit.csail.mit.edu/papers/2017/kjolstad-oopsla17-tensor-compiler.pdf)

## Core IR Design

**Three key abstractions:**

```
1. Tensor (with format metadata)
├── name, shape [rows, cols]
├── format: 'CSR' or 'CSC'
├── features: dict{'density': float, 'nnz_clustering': float, 'row_span': float, 
                  'col_span': float, 'bandwidth': float, 'irregularity': float}
└── iter_spaces(): returns ['outer', 'nnz'] where:
    CSR: outer=row_idx (0..rows), nnz=row_ptr[r]..row_ptr[r+1]
    CSC: outer=col_idx (0..cols), nnz=col_ptr[c]..col_ptr[c+1]
```

```
2. IndexVar
├── name: 'i', 'j', 'k' (logical indices)
├── dim: tensor dimension (0=row, 1=col)
└── physical_mapping: maps to iter_space ('outer', 'nnz')
```

```
3. SparseLoopNest (your main IR)
├── loops: List[Loop]  # nested from outer→inner
├── body: Assign | Reduce  # y(i) = ..., y(i) += ...
└── tensor_accesses: maps tensors to their IndexVars used
```

```
Loop:
├── idxvar: IndexVar
├── bounds: (lower, upper)  # symbolic: 'A.shape[0]', 'row_ptr[r+1]'
├── body: SparseLoopNest | Assign | Reduce
├── tiled: bool (post-tiling flag)
└── reordered: bool
```

## DSL → IR Lowering (Generic)

From your DSL `y(i) = A(i,j) * x(j)` (SpMV):

1. **Parse to AST** → tensor vars + index equation `i==A.i, j==A.j, j==x.j`
2. **Lower to iteration order**:  
   - Pick canonical order based on format: CSR=`A.row_idx → A.nnz_idx`, CSC=`A.col_idx → A.nnz_idx`
   - Generate nested loops matching `A.iter_spaces()`
3. **Populate body**: `y.outer += A.nnz * x[A.nnz_idx.j]`

Same lowering works for **all kernels**:
- **SpMM**: `Y(i,k) = A(i,j) * B(j,k)` → outer=i, mid=nnz_j, inner=k (if fused)
- **SpGEMM**: Similar but with symbolic nnz accumulation  
- **SpAdd/SpElMul**: Pointwise over nnz with merge logic
- **SDDMM**: Dense outer + sparse inner

## Optimisation Passes (Operate purely on LoopNest)

**Pass 1: Blocking (Loop Tiling)**
```
def tile_loop(loop, tile_size):
    # Split outer loop: for r=0; r<rows; r+=tile_size
    #                   for rt=0; rt<tile_size; rt++
    #                     for r_inner = r+rt  (adjust bounds)
    # Only tile outer/row loops where CSR bandwidth fits tile
    pass
```

**Pass 2: Interchange**
```
def interchange_loops(nest):
    # Swap outer↔nnz if clustering suggests it improves locality
    # Verify dependencies: no data races on accumulators
    pass
```

**Correctness via invariants** (enforced in lowering + passes):
1. **Bounds safety**: nnz_idx always between row_ptr[r] and row_ptr[r+1]
2. **Index consistency**: logical j from A.nnz_idx.j always valid col
3. **Full coverage**: tiled loops cover exact same iteration space
4. **No overlap**: reordering preserves dependence order

## Codegen (IR → C)

```
def emit_c(nest):
    for loop in nest.loops:
        emit_for_loop(loop.idxvar, loop.bounds)  # handles symbolic bounds
        emit_c(loop.body)
    
    # In body: y[outer] += A.vals[nnz_idx] * x[A.col_idx[nnz_idx]]
```

## Refactor Steps (Minimal Changes)

1. **Wrap your current IR** as `SparseLoopNest` class/tree.
2. **Add format metadata** to tensors (`CSR/CSC` + `iter_spaces()` method).
3. **Generalise kernel lowering**: write a `lower_algebra_to_iteration(dsl_ast, format)` that generates the loop nest for *any* kernel matching your patterns.
4. **Passes read/write flags** on loops (`needs_tiling=True`, `tile_size=32`).

This refactor is **~1–2 weeks of work** if your current codegen already emits reasonable C loops, and makes your system genuinely generative (DSL → any kernel → optimised C) rather than kernel‑specific.

Your heuristics then become pass flags that control *whether* to tile/reorder the generically‑generated nests based on features. Much stronger story for evaluation!