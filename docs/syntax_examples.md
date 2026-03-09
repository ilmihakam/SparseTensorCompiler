# DSL Syntax Examples

## Basic Tensor Operations

### Matrix-Vector Multiplication
```
// Declaration
A: tensor[i, j];
x: tensor[j];
y: tensor[i];

// Computation (Einstein notation - sum over j)
compute y[i] = A[i, j] * x[j];
```

### Matrix-Matrix Multiplication
```
A: tensor[i, k];
B: tensor[k, j];
C: tensor[i, j];

// Standard matrix multiplication
compute C[i, j] = A[i, k] * B[k, j];
```

### Element-wise Operations
```
A: tensor[i, j];
B: tensor[i, j];
C: tensor[i, j];

// Addition
compute C[i, j] = A[i, j] + B[i, j];

// Hadamard product
compute C[i, j] = A[i, j] * B[i, j];

// Scalar multiplication
compute C[i, j] = 2.5 * A[i, j];
```

## Machine Learning Operations

### Neural Network Layer
```
// Weight matrix and bias vector
W: tensor[i, j];
b: tensor[i];
x: tensor[j];
y: tensor[i];

// Linear transformation
compute y[i] = W[i, j] * x[j] + b[i];

// With activation function
compute y[i] = relu(W[i, j] * x[j] + b[i]);
```

### Convolution-like Operations
```
// Input tensor and kernel
input: tensor[i, j, k];
kernel: tensor[m, n];
output: tensor[i, j];

// 2D convolution (simplified)
compute output[i, j] = input[i+m, j+n, k] * kernel[m, n];
```

### Attention Mechanism
```
Q: tensor[i, d];    // Query
K: tensor[j, d];    // Key
V: tensor[j, d];    // Value
A: tensor[i, j];    // Attention weights
O: tensor[i, d];    // Output

// Attention computation
compute A[i, j] = softmax(Q[i, k] * K[j, k]);
compute O[i, d] = A[i, j] * V[j, d];
```

## Complex Expressions

### Multi-layer Operations
```
// Layer 1
compute h1[i] = relu(W1[i, j] * x[j] + b1[i]);

// Layer 2
compute h2[i] = relu(W2[i, j] * h1[j] + b2[i]);

// Output layer
compute y[i] = softmax(W3[i, j] * h2[j] + b3[i]);
```

### Tensor Contractions
```
// Higher-order tensor operations
A: tensor[i, j, k];
B: tensor[k, l, m];
C: tensor[i, j, l, m];

// Tensor contraction over k
compute C[i, j, l, m] = A[i, j, k] * B[k, l, m];
```

### Mixed Operations
```
// Residual connection
input: tensor[i];
W1: tensor[i, j];
W2: tensor[j, i];
output: tensor[i];

compute temp[j] = relu(W1[i, j] * input[i]);
compute output[i] = input[i] + W2[j, i] * temp[j];
```

## Function Integration Examples

### Custom Activation Functions
```
// Using custom functions
compute y[i] = gelu(W[i, j] * x[j] + b[i]);
compute y[i] = swish(x[i], 1.0);
compute y[i] = layer_norm(x[i], scale[i], bias[i]);
```

### Reduction Operations
```
// Using reduction functions (alternative to Einstein notation)
compute mean_val = mean(A[i, j]);
compute max_val = max(A[i, j]);
compute sum_rows[i] = sum_j(A[i, j]);
```

### Conditional Operations
```
// Conditional execution
compute y[i] = select(mask[i], A[i, j] * x[j], 0.0);
compute y[i] = clip(A[i, j] * x[j], -1.0, 1.0);
```

## Index Patterns

### Free vs Bound Indices
```
// Free indices: i, j (appear on left side)
// Bound indices: k (summed over, only on right side)
compute C[i, j] = A[i, k] * B[k, j];

// Multiple bound indices
compute result[i] = A[i, j, k] * B[j, k];
```

### Broadcasting Patterns
```
// Scalar to tensor
compute C[i, j] = A[i, j] + scalar_val;

// Vector to matrix (broadcast)
compute C[i, j] = A[i, j] + b[j];
```

### Index Reuse
```
// Same index used multiple times
compute trace = A[i, i];  // Matrix trace
compute diag[i] = A[i, i] * B[i, i];  // Element-wise diagonal product
```

## Error Examples

### Syntax Errors
```
// Missing semicolon
compute y[i] = A[i, j] * x[j]  // ERROR

// Unmatched brackets
compute y[i] = A[i, j * x[j];  // ERROR

// Invalid identifier
compute 2y[i] = A[i, j] * x[j];  // ERROR
```

### Semantic Errors
```
// Index mismatch
A: tensor[i, j];
B: tensor[k, l];
compute C[i, j] = A[i, j] + B[k, l];  // ERROR: incompatible indices

// Undefined tensor
compute y[i] = undefined_tensor[i, j] * x[j];  // ERROR

// Invalid Einstein notation
compute C[i, j] = A[i, k] * B[l, m];  // ERROR: no shared index
```