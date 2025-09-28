
# Honours Project Overview
Machine learning matrices are filled with near-zero or zero values, which has prompted exploration into sparse matrices that avoid computing on these values. However, unlike their dense equivalents, sparse tensors struggle to take advantage of modern hardware, resulting in massive inefficiencies.

Compiler infrastructure struggles to fill this gap, as each different format requires a different set of optimization techniques.  In this project, you will look at optimization strategies for common operations on these data structures.

In this project, you will pick an existing, underexplored sparse format and optimize it. The key outcomes will be the following:
  - Build simple matrix operations (add, multiply) in your sparse format
  - Optimise those matrix operations by hand
  - Build a simple compiler infrastructure that generates these optimised matrix
     operations (students looking for an easy project should end here)
  - Build a simple optimiser for sequences of these operations to enable operator fusion on your datastructure of choice (see suggestions below).

Data structure choice is left up to you, but the following suggestions would make good projects: COO (coordinate list), CSR (compressed sparse row), CSC (compressed sparse column), ELL (ELLPACK), and DIA (Diagonal format)  --- more suggestions can be found in section 2.1.3 of https://dl.acm.org/doi/pdf/10.1145/3571157 and I'm open to other suggestions for formats.

# Recommended Reading

Format Abstraction for Sparse Tensor Compilers: https://dl.acm.org/doi/10.1145/3276493 
A Sparse Iteration Space Transformation Framework for Sparse Tensor Algebra - https://dl.acm.org/doi/10.1145/3428226
The Sparse Tensor Algebra - https://dl.acm.org/doi/10.1145/3133901
Sparse Tensor Algebra Optimization - https://arxiv.org/pdf/1802.10574

# Sparse Tensor Base Format
<TensorName>: <Type>, [i, j, k]  (type in in CSC, COO, ELLPACK, CSR)
for [<TensorNameList>] [<Iterator Index>] {
compute A[i][j] = B[i][j] * C[i][j] + D[i]
call <Function>(<TensorName>)
}
