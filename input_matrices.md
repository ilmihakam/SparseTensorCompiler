Matrix Dimension

Size of CSR/CSC index arrays and value arrays;

Number of distinct elements of the input vector x and output vector y touched each SpMV.



NNZ Number

''

Sparsity Density

(Measure by nnz/row or nnz/col)

Very sparse (nnz small): dominated by compulsory misses; sometimes blocking does not help much because there is little reuse.​

Less sparse (denser but still sparse): more opportunities to reuse x elements and matrix blocks in cache, if structure is favorable.

These control regularity and how well you can map to SIMD / blocked formats and balance work.



Average nnz/row, nnz/column

High average nnz/row has: 

longer inner loops (CSR-style), better amortization of index overhead;

more chances to reuse nearby x[j] within a single row.

Low average nnz/row has

short loops, poor vectorization;

overhead from index loads dominates.



Variance/Irregularity of nnz

Low variance (rows all about the same length): good for formats like ELL/SELL-C-σ and GPU vectorization; regular access pattern; predictable prefetch.

High variance: some very long and many short rows:

poor load balance (for parallel implementations);

each warp/core sees inconsistent access lengths, hurting cache-line utilization and prefetch efficiency.

Matrix Bandwidth and

profile 

Standard definitions:

Bandwidth: ∣i−j∣maxaij=0∣i−j∣

Profile: sum over rows (or columns) of how far nnz extend from the diagonal.

These are explicitly used as locality metrics in performance models for SpMV, alongside sparsity and nnz/row/col.



NNZ Clustering and Block Structure

Block-diagonal or banded structure:

NNZ concentrated near diagonal or in a few diagonals → good temporal reuse of x and y locally.

Small dense blocks (block structure):

Regions where nnz form dense r×cr×c blocks support BCSR, blocked ELL, register blocking, etc.

Performance depends strongly on an “effective block size” that yields high block occupancy (many nnz per block).

Manhattan Distance/Locality Metrics

Low average Manhattan distance between consecutive nnz → better spatial locality in both matrix arrays and x.

High value → “jumping around” in x and in matrix index/value arrays.

Spatial Locality of column indices within rows

Fine-grained version of bandwidth/column-span:

Look at the differences jk+1−jk between sorted column indices within each row.

Metrics: average stride, distribution of strides, fraction of stride-1 or small-stride transitions.

Temporal Reuse distance of x entries

Short reuse distance (same j or nearby j used again soon, e.g., in adjacent rows):

higher chance x[j] remains in L1/L2 and in the TLB.

Long reuse distance:

even if x fits in memory comfortably, you may evict its cache lines or TLB entries before reuse.

Inter-row similarity

Jaccard similarity or intersection size between column sets of adjacent or nearby rows.

Graph/hypergraph partitioning measures (see below) that cluster rows with shared columns.

Irregularity and Randomness of Sparse Pattern

Irregular (e.g., graph-like) matrices typically show:

Power-law degree distribution in the column/row graph.

High entropy in column indices within rows.

Models sometimes assume uniform distribution of nonzeros to derive probabilistic bounds on miss rates; deviations from this model (clustering, skew) change how accurate such predictions are.