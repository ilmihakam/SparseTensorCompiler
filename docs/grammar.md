# Sparse Tensor DSL Grammar Specification

## Design Goals
- **Type-Enforced**: Explicit tensor type declarations (Dense, CSR, COO, CSC, ELLPACK, DIA)
- **Shape-Aware**: Optional shape specifications for compile-time validation
- **Einstein Notation**: Support implicit summation over repeated indices
- **ML Integration**: Support function calls for ML models
- **Clean Syntax**: Readable and mathematically intuitive

## Complete Grammar (Extended BNF)

### Program Structure
```ebnf
program         ::= statement*

statement       ::= declaration ';'
                  | computation ';'
                  | call_statement ';'
                  | for_statement
```

### Tensor Declarations (Type-Enforced)
```ebnf
declaration     ::= 'tensor' IDENTIFIER ':' tensor_type ('<' shape_list '>')?

tensor_type     ::= 'Dense' | 'CSR' | 'CSC'

shape_list      ::= NUMBER (',' NUMBER)*
```

**Examples:**
- `tensor A : CSR<100, 50>;` - CSR sparse matrix, 100x50
- `tensor B : Dense<50, 20>;` - Dense matrix, 50x20
- `tensor C : COO;` - COO format, shape to be inferred

### Statements
```ebnf
computation     ::= 'compute' tensor_access '=' expression

call_statement  ::= 'call' IDENTIFIER '(' argument_list? ')'

for_statement   ::= 'for' '[' identifier_list ']' '[' index_list ']'
                         '{' statement_list '}'

statement_list  ::= statement+
```

### Expressions
```ebnf
expression      ::= tensor_access
                  | function_call
                  | NUMBER
                  | IDENTIFIER
                  | expression '+' expression
                  | expression '*' expression
                  | '(' expression ')'

tensor_access   ::= IDENTIFIER '[' index_list ']'

function_call   ::= IDENTIFIER '(' argument_list? ')'

argument_list   ::= expression (',' expression)*
```

### Lists
```ebnf
index_list      ::= IDENTIFIER (',' IDENTIFIER)*

identifier_list ::= IDENTIFIER (',' IDENTIFIER)*
```

### Lexical Rules
```ebnf
IDENTIFIER      ::= [a-zA-Z_][a-zA-Z0-9_]*
NUMBER          ::= [0-9]+('.'[0-9]+)?
```

### Operator Precedence (Highest to Lowest)
1. `*` (multiplication) - left associative
2. `+` (addition) - left associative

## Syntax Examples

### Basic Operations
```
// Tensor declarations (type-enforced)
tensor A : CSR<100, 50>;
tensor B : Dense<50, 20>;
tensor C : Dense<100, 20>;

// Matrix multiplication (Einstein notation - implicit sum over k)
compute C[i, j] = A[i, k] * B[k, j];

// Element-wise operations
tensor X : Dense<10, 10>;
tensor Y : Dense<10, 10>;
compute Y[i, j] = X[i, j] + X[i, j];

// Scalar operations
tensor result : Dense<10, 10>;
compute result[i, j] = 2.0 * X[i, j];
```

### Function Integration
```
// ML model function calls
tensor A : Dense<10, 10>;
tensor B : Dense<10, 10>;
tensor result : Dense<10, 10>;

compute result[i, j] = relu(A[i, j] * B[i, j]);
compute result[i, j] = f(A[i, j], B[i, j], 0.5);

// Chained operations
tensor temp : Dense<100, 20>;
tensor bias : Dense<100, 20>;
compute temp[i, j] = A[i, k] * B[k, j];
compute result[i, j] = relu(temp[i, j] + bias[i, j]);
```

## Semantic Rules

### Index Variables
- **Repeated indices**: Implicit summation (Einstein notation)
- **Free indices**: Must appear on both sides of assignment
- **Bound indices**: Appear only on right side (summation variables)

Example:
```
compute C[i, j] = A[i, k] * B[k, j];
// i, j are free indices (appear on left side)
// k is bound index (summed over, only on right side)
```

### Type Checking
- All tensors in expression must have compatible dimensions
- Function arguments must match expected types
- Index consistency across tensor accesses

### Format Specification
- Tensor format is explicitly declared in the type system
- Supported formats: Dense, CSR, CSC
- Format selection enables format-specific optimizations during compilation
- Shape information (optional) enables compile-time validation

## Error Handling

### Syntax Errors
- Malformed expressions
- Missing semicolons
- Unmatched parentheses
- Invalid identifiers

### Semantic Errors
- Undefined tensors/functions
- Index dimension mismatches
- Type incompatibilities
- Invalid Einstein notation

## Implementation Notes

### Lexical Analysis
- Keywords: `tensor`, `compute`
- Operators: `+`, `-`, `*`, `/`, `=`
- Delimiters: `[`, `]`, `(`, `)`, `,`, `;`, `:`
- Identifiers: Variable and function names
- Literals: Numeric constants

### Parsing Strategy
- Recursive descent parser
- Precedence climbing for expressions
- Left-to-right associativity for same precedence
- Error recovery at statement boundaries

### AST Design
- Format-agnostic representation
- Index variables as first-class entities
- Function calls as expression nodes
- Einstein notation metadata