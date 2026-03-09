Given your IR, codegen should become “render scheduled IR + low-level `ir` statements to C” instead of “emit hard-coded strings per kernel”. Concretely:

***

## 1. What codegen should take as input

Use **scheduled IR** as the single entry-point:

- At the top level: `scheduled::Program` → iterate its `scheduled::Compute` nodes.
- For each compute: use:
  - `scheduled::Compute::rootLoop` (the loop tree),
  - its `prologueStmts`, `epilogueStmts`,
  - and the `ir::IRStmt`/`ir::IRExpr` nodes in `preStmts`/`postStmts` and leaf bodies. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_958d8485-7863-4b3d-9ead-09279ae74e25/9d9d37d3-1d1b-4e9e-bae1-373dd5813707/Sparse-Tensor-Algebra-Framework.pdf)

You should not need to know “this is SpMV vs SpMM vs intersection”: all of that is encoded in the loop kinds, bounds, merge strategies, and the IR statements the scheduler built. [commit.csail.mit](https://commit.csail.mit.edu/papers/2017/kjolstad-oopsla17-tensor-compiler.pdf)

***

## 2. Overall structure of the new codegen

Refactor into a **generic renderer**:

```text
generate_c(program):
  emit headers / typedefs
  for each scheduled::Declaration:
    emit tensor structs / globals
  for each scheduled::Compute:
    emit_compute_function(compute)
```

```text
emit_compute_function(compute):
  emit "void compute_<name>(...tensors...) {"
    emit_ir_block(compute.prologueStmts, indent=1)
    emit_loop_tree(compute.rootLoop, indent=1)
    emit_ir_block(compute.epilogueStmts, indent=1)
  emit "}"
```

Where:

- `emit_loop_tree` walks `scheduled::Loop` recursively.
- `emit_ir_block` is a generic printer for low-level `ir::IRStmt` (Assign, VarDecl, If, For, RawStmt, etc.). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_958d8485-7863-4b3d-9ead-09279ae74e25/9d9d37d3-1d1b-4e9e-bae1-373dd5813707/Sparse-Tensor-Algebra-Framework.pdf)

***

## 3. Rendering scheduled loops

Use the scheduling metadata already present:

Each `scheduled::Loop` has:

- `indexName`, `lower`, `upper`, `runtimeBound`
- `LoopKind` (`Dense`, `Sparse`, `Block`)
- `driverTensor`, `parentIndexOverride`
- `preStmts`, `postStmts`
- `children` [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_958d8485-7863-4b3d-9ead-09279ae74e25/9d9d37d3-1d1b-4e9e-bae1-373dd5813707/Sparse-Tensor-Algebra-Framework.pdf)

So `emit_loop_tree` looks like:

```pseudo
emit_loop_tree(loop, indent):
  # 1. Emit loop header
  if loop.kind == Dense:
    header = f"for (int {loop.indexName} = {emit_expr(loop.lower)}; " +
             f"{loop.indexName} < {emit_expr(loop.upper)}; ++{loop.indexName}) {{"
  elif loop.kind == Sparse:
    # CSR/CSC-specific: use driverTensor + parentIndexOverride to choose row_ptr/col_ptr
    header = emit_sparse_loop_header(loop)
  elif loop.kind == Block:
    header = emit_block_loop_header(loop)  # outer tile loop

  print INDENT(indent) + header

  # 2. Emit pre-statements
  emit_ir_block(loop.preStmts, indent+1)

  # 3. Emit children or leaf body
  if loop.children not empty:
    for child in loop.children:
      emit_loop_tree(child, indent+1)
  else:
    emit_ir_block(loop.bodyStmts, indent+1)  # whatever the scheduler put at the leaf

  # 4. Emit post-statements
  emit_ir_block(loop.postStmts, indent+1)

  print INDENT(indent) + "}"
```

The **CSC/CSR switch** that your current kernel code hard‑codes becomes a generic `emit_sparse_loop_header(loop)` that reads:

- `loop.driverTensor.format` (`CSR` or `CSC`),
- the parent index (row vs col) via `parentIndexOverride`,
- and chooses between row_ptr/col_ptr accordingly. [fredrikbk](http://fredrikbk.com/publications/taco-scheduling.pdf)

***

## 4. Rendering `ir::IRStmt` / `ir::IRExpr`

You already have a low-level structured IR (`IRTensorAccess`, `IRConstant`, `IRBinaryOp`, `IRAssign`, `IRIfStmt`, `IRForStmt`, `IRRawStmt`, etc.). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_958d8485-7863-4b3d-9ead-09279ae74e25/9d9d37d3-1d1b-4e9e-bae1-373dd5813707/Sparse-Tensor-Algebra-Framework.pdf)

Write **one printer** for these:

```pseudo
emit_ir_block(stmts, indent):
  for stmt in stmts:
    emit_ir_stmt(stmt, indent)

emit_ir_stmt(stmt, indent):
  switch stmt.kind:
    case VarDecl:
      print INDENT(indent) + type + " " + name + ";"
    case Assign:
      print INDENT(indent) + emit_expr(stmt.lhs) + " = " + emit_expr(stmt.rhs) + ";"
    case IfStmt:
      print INDENT(indent) + "if (" + emit_expr(stmt.cond) + ") {"
      emit_ir_block(stmt.thenStmts, indent+1)
      if stmt.elseStmts not empty:
        print INDENT(indent) + "} else {"
        emit_ir_block(stmt.elseStmts, indent+1)
      print INDENT(indent) + "}"
    case ForStmt:
      # nested low-level for (used e.g. in postStmts)
      print INDENT(indent) + "for (" + ... + ") {"
      emit_ir_block(stmt.body, indent+1)
      print INDENT(indent) + "}"
    case RawStmt:
      print INDENT(indent) + stmt.rawText
```

`emit_expr` similarly handles `IRConstant`, `IRBinaryOp`, `IRTensorAccess`, etc. `IRTensorAccess` is where you map tensor metadata + indices to actual C array accesses (`vals[p]`, `row_idx[p]`, etc.). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_958d8485-7863-4b3d-9ead-09279ae74e25/9d9d37d3-1d1b-4e9e-bae1-373dd5813707/Sparse-Tensor-Algebra-Framework.pdf)

***

## 5. Where your current kernel‑specific code fits

Your current `emitIntersectionSparseReference()` function is expressing a particular **scheduled loop tree**:

- Outer dense loop over `j` (CSC) or `i` (CSR).
- Inner loop over `pC` driven by C’s structure.
- Scalar temporaries (`pA`, `pB`, `endA`, `endB`, `a_val`, `b_val`).
- `while` loops to advance `pA`/`pB`.
- Final assignment `C_ref_vals[pC] = a_val * b_val;`.

In the new design, you should:

1. Express this algorithm as **scheduled IR + low-level IR**, not as C strings:

   - `scheduled::Loop` for outer and inner loops, with `driverTensor = C`, `LoopKind::Sparse`, `mergedTensors = {A,B}`, `mergeStrategy = Intersection` for the inner loop. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_958d8485-7863-4b3d-9ead-09279ae74e25/9d9d37d3-1d1b-4e9e-bae1-373dd5813707/Sparse-Tensor-Algebra-Framework.pdf)
   - Put declarations of `pA`, `pB`, `endA`, `endB`, `a_val`, `b_val` into `preStmts` of the outer loop and/or inner loop.  
   - Put the `while` statements and final assignment into `loop.bodyStmts` using `IRWhileStmt` / `IRAssign` / `IRCompareExpr`.  

2. Let generic codegen emit it:

   - `scheduleCompute` constructs the `Loop` tree + `preStmts`/`bodyStmts`.  
   - `emit_compute_function` → `emit_loop_tree` → `emit_ir_block` prints the exact C structure.

The CSC vs CSR split becomes part of scheduling (`driverTensor` + bounds expressions), not part of codegen.

***

## 6. Refactor steps

1. **Stop calling kernel‑specific emitters** in CodeGenerator.  
   Instead, after scheduling, always call:

   ```cpp
   void CodeGenerator::emitCompute(const scheduled::Compute &c) {
       emitFunctionHeader(c);
       emit_ir_block(c.prologueStmts, 1);
       emit_loop_tree(*c.rootLoop, 1);
       emit_ir_block(c.epilogueStmts, 1);
       emitFunctionFooter();
   }
   ```

2. **Implement `emit_loop_tree`** based on `LoopKind`, `driverTensor.format`, and `parentIndexOverride` as above.

3. **Implement `emit_ir_stmt/emit_expr`** to print the low-level `ir` nodes you already defined, and migrate any `IRRawStmt` string fragments into structured nodes where possible.

4. **Gradually remove kernel‑specific functions** like `emitIntersectionSparseReference` by moving their logic into the scheduler (building appropriate loop trees and IR statements) instead of directly emitting C.

***

This refactor makes your codegen:

- **Kernel-agnostic**: any new kernel that lowers to `scheduled::Compute` automatically gets C code.  
- **Format-aware but modular**: CSR vs CSC is chosen in scheduling/loop bounds, not hard‑wired into emitters.  
- **Compatible with your optimisation work**: blocking / reordering just rewrite `scheduled::Loop` trees; codegen remains unchanged.