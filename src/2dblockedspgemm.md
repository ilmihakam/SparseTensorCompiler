How 2D blocking works for SpGEMM
SpGEMM (C = A × B, both sparse) is the most complex of your kernels because it’s not just “follow one sparsity pattern” like SpMV—it’s a sparse outer product where each nnz in A “touches” a whole row/column segment of B. This makes it a prime candidate for 2D blocking.

Core idea of 2D blocked SpGEMM
Tile the iteration space into 2D blocks:

Partition A into row-blocks and column-blocks: 
A
i
,
j
A 
i,j
  where 
i
i indexes row-tiles, 
j
j indexes column-tiles.

Partition B into row-blocks and column-blocks: 
B
k
,
l
B 
k,l
 .

Each output tile 
C
p
,
q
C 
p,q
  depends on a rectangular slab:

text
C_{p,q} = sum over k of (A_{p,k} × B_{k,q})
Where 
A
p
,
k
A 
p,k
  is a 2D sparse tile from A (row-block 
p
p, column-block 
k
k).

B
k
,
q
B 
k,q
  is a 2D sparse tile from B (row-block 
k
k, column-block 
q
q).

Compute each output tile:

Load the relevant input tiles into cache/shared memory.

For each nnz in 
A
p
,
k
A 
p,k
 , multiply with the matching row segment of 
B
k
,
q
B 
k,q
  (via merge/intersection on indices).

Accumulate into a local dense tile buffer for 
C
p
,
q
C 
p,q
  (or sparse accumulator if you want).

Example with 16×16 tiles (CSR format)
Assume A and B are CSR. Pseudocode sketch:

text
for each output row-tile p:
  for each output column-tile q:
    # Allocate local dense accumulator for C[p,q] (16x16 = 256 entries)
    initialize C_tile[16][16] = 0
    
    # Loop over contributing k (input column-tiles of A / row-tiles of B)
    for k in 0 .. num_col_tiles_A:
      # Load sparse tile A[p,k]: ~CSR within the 16x16 block
      # (row_ptrs, col_idxs, vals clipped to this tile)
      A_tile = load_sparse_tile(A, row_start=p*16, col_start=k*16)
      
      # Load sparse tile B[k,q]: ~CSR within the 16x16 block
      B_tile = load_sparse_tile(B, row_start=k*16, col_start=q*16)
      
      # Symbolic/numeric multiply-accumulate:
      for each nnz (i,j,val) in A_tile:
        for each nnz (j',val') in B_tile.row(j):  # Match on column of A == row of B
          if j == j':
            C_tile[i][j'] += val * val'  # Accumulate in dense tile
    
    # Scatter dense C_tile back to global C[p,q], pruning zeros if desired
    scatter_sparse(C, C_tile)
Key benefits of 2D blocking for SpGEMM
Cache reuse:

A
p
,
k
A 
p,k
  and 
B
k
,
q
B 
k,q
  tiles are small (e.g., 16×16 sparse = ~few hundred bytes), so they fit entirely in L1/L2.

Multiple nnz in 
A
p
,
k
A 
p,k
  reuse the same segments of 
B
k
,
q
B 
k,q
 .

Accumulator locality:

C
p
,
q
C 
p,q
  accumulator lives entirely in registers/shared memory—no global writes until the full tile is computed.

Prefetching:

Within-tile accesses are more regular (CSR within small bounds).

Challenges and your format fit
Your kernels use CSR (row-major), so:

Outer loops: natural row-block partitioning (p over row-tiles).

Inner column tiles (q, k): require either:

Precomputing column-block metadata (e.g., pos arrays per column-tile).

Or dynamic extraction: when processing row 
i
i, clip column indices to the current q-range.

In practice, many SpGEMM implementations (e.g., TileSpGEMM) use a hybrid sparse tile format within each 2D block: CSR + column index clipping + bitmasks for fast intersection.
​

For your compiler-generated C code, a simpler version is fine:

Stick to CSR for input tiles.

Extract relevant segments on-the-fly using binary search or linear scan on pos/crd arrays.

You saw blockonly helping on large SpGEMM matrices like “cant” (1.05–1.14 speedup), which likely benefits from exactly this tile-level reuse.
​

Pseudocode for your compiler
Extend your existing iteration graph + merge lattice to support 2D tiles:

text
# Outer iteration graph now over tile indices (p, q)
for p in 0 .. num_row_tiles:
  for q in 0 .. num_col_tiles:
    # Nested loops now scoped to tile bounds
    for i in p*Ti .. (p+1)*Ti:      # Tile-local row
      for pA in A_pos[i]:           # nnz in row i
        jA = A_crd[pA]
        if jA in tile q:            # Only nnz contributing to this col-tile
          # Merge with matching rows in B tile (k,q)
          # Accumulate into local C_tile[i % Ti][jA % Tq]
This fits your sparse iteration theory nicely—your provenance graphs already track coordinate mappings, so you can map tile-local i/j back to global.